from __future__ import annotations

import copy
import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import pas

DTYPE = torch.float64


class _LifecycleNoise(Mechanism):
    Mechanism.RANGENOISE("eta", distribution="normal", mu=0.0, sigma=1.0, seed=2468)

    def i(self, v):
        return self.eta * 1.0e-4


def _population(*, noisy=False):
    pop = dn.SingleCompartment(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    pop.insert(pas, g=0.001, e=-70.0)
    if noisy:
        pop.insert(_LifecycleNoise)
    pop.initialize()
    return pop


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        cloned = value.__class__(
            (key, _clone_nested(item)) for key, item in value.items()
        )
        if hasattr(value, "_metadata"):
            cloned._metadata = copy.deepcopy(value._metadata)
        return cloned
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return copy.deepcopy(value)


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


@pytest.mark.parametrize("invalid", [0.0, -0.1, math.inf, math.nan, True])
@pytest.mark.parametrize(
    "entrypoint",
    ["step", "run", "longrun", "checkpointed", "steady_state"],
)
def test_population_time_entrypoints_reject_invalid_dt_without_mutation(
    entrypoint, invalid
):
    pop = _population()
    before = _clone_nested(pop.state_dict_for_checkpoint())
    cache_names = set(pop._caches)

    with pytest.raises((TypeError, ValueError), match="dt"):
        if entrypoint == "step":
            pop.step(dt=invalid)
        elif entrypoint == "run":
            pop.run(tstop=0.02, dt=invalid)
        elif entrypoint == "longrun":
            pop.longrun(tstop=0.02, chunklength=1, dt=invalid)
        elif entrypoint == "checkpointed":
            pop.longrun_checkpointed(tstop=0.02, chunklength=1, dt=invalid)
        else:
            pop.steady_state(tstop=0.02, dt=invalid)

    _assert_nested_equal(pop.state_dict_for_checkpoint(), before)
    assert set(pop._caches) == cache_names


@pytest.mark.parametrize("invalid", [-0.1, math.inf, math.nan, True])
@pytest.mark.parametrize(
    "entrypoint", ["run", "longrun", "checkpointed", "steady_state"]
)
def test_population_time_entrypoints_reject_invalid_tstop_without_mutation(
    entrypoint, invalid
):
    pop = _population()
    before = _clone_nested(pop.state_dict_for_checkpoint())
    cache_names = set(pop._caches)

    with pytest.raises((TypeError, ValueError), match="tstop"):
        if entrypoint == "run":
            pop.run(tstop=invalid, dt=0.01)
        elif entrypoint == "longrun":
            pop.longrun(tstop=invalid, chunklength=1, dt=0.01)
        elif entrypoint == "checkpointed":
            pop.longrun_checkpointed(tstop=invalid, chunklength=1, dt=0.01)
        else:
            pop.steady_state(tstop=invalid, dt=0.01)

    _assert_nested_equal(pop.state_dict_for_checkpoint(), before)
    assert set(pop._caches) == cache_names


def test_failed_population_initialize_is_fail_closed_and_retryable():
    pop = _population()
    pop.step(dt=0.01)
    assert pop.integrator.initialized

    def fail(_model):
        raise RuntimeError("injected initialize failure")

    pop.register_pre_initialize_hook(fail)
    with pytest.raises(RuntimeError, match="injected initialize failure"):
        pop.initialize()

    assert not pop.initialized
    assert not pop.integrator.initialized
    with pytest.raises(ValueError, match="initialized"):
        pop.step(dt=0.01)

    pop.pre_initialize_hooks.clear()
    assert pop.initialize() is pop
    assert pop.initialized
    assert not pop.integrator.initialized
    pop.step(dt=0.01)
    assert pop.integrator.initialized


def test_failed_cached_initialize_post_hook_remains_fail_closed_and_retryable():
    pop = _population()
    pop.cache("_steady_state")

    def fail(_model):
        raise RuntimeError("injected cached initialize failure")

    pop.register_post_initialize_hook(fail)
    with pytest.raises(RuntimeError, match="cached initialize failure"):
        pop.initialize()

    assert not pop.initialized
    assert not pop.initializing_from_state_cache
    assert not pop._restoring_steady_state
    assert not pop.integrator.initialized
    with pytest.raises(ValueError, match="initialized"):
        pop.step(dt=0.01)

    pop.post_initialize_hooks.clear()
    assert pop.initialize() is pop
    assert pop.initialized
    assert pop.initializing_from_state_cache


def test_named_cache_restore_is_reusable_and_replays_rng_suffix():
    pop = _population(noisy=True)
    pop.step(dt=0.01)
    pop.cache("boundary")

    pop.step(dt=0.01)
    expected_v = pop.v.clone()
    expected_eta = pop.mech._LifecycleNoise.eta.clone()
    expected_t = pop.t.clone()

    for _ in range(2):
        assert pop.restore("boundary") is pop
        pop.step(dt=0.01)
        torch.testing.assert_close(pop.v, expected_v)
        torch.testing.assert_close(pop.mech._LifecycleNoise.eta, expected_eta)
        torch.testing.assert_close(pop.t, expected_t)


def test_pre_step_cache_survives_lazy_solver_workspace_initialization():
    pop = _population()
    pop.cache("before-first-step")

    pop.step(dt=0.01)
    expected_v = pop.v.clone()
    expected_t = pop.t.clone()

    assert pop.restore("before-first-step") is pop
    assert not pop.integrator.initialized
    pop.step(dt=0.01)

    torch.testing.assert_close(pop.v, expected_v)
    torch.testing.assert_close(pop.t, expected_t)


def test_corrupt_named_cache_rolls_back_all_live_state_atomically():
    pop = _population(noisy=True)
    pop.step(dt=0.01)
    pop.cache("corrupt")
    pop.step(dt=0.01)
    before = _clone_nested(pop.state_dict())

    cached = pop._caches["corrupt"]
    cached["v"].fill_(-12.0)
    rng_key = next(key for key in cached if key.endswith("eta_rng._extra_state"))
    cached[rng_key] = {}

    with pytest.raises(KeyError, match="base_seed.*rng_state"):
        pop.restore("corrupt")

    _assert_nested_equal(pop.state_dict(), before)


def test_named_cache_rejects_unexpected_workspace_keys_atomically():
    pop = _population()
    pop.step(dt=0.01)
    pop.cache("unexpected")
    before = _clone_nested(pop.state_dict())
    pop._caches["unexpected"]["integrator.unexpected_workspace"] = torch.tensor(1.0)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        pop.restore("unexpected")

    _assert_nested_equal(pop.state_dict(), before)


def test_pre_batch_cache_shape_failure_is_atomic():
    pop = _population()
    pop.cache("unbatched")
    pop.batch(2)
    before = _clone_nested(pop.state_dict())

    with pytest.raises(RuntimeError, match="size mismatch"):
        pop.restore("unbatched")

    _assert_nested_equal(pop.state_dict(), before)


@given(
    st.lists(
        st.sampled_from(
            ["step", "cache", "perturb", "restore", "initialize", "float", "double"]
        ),
        min_size=1,
        max_size=14,
    )
)
@settings(max_examples=30, deadline=None)
def test_population_lifecycle_operation_sequences_preserve_contracts(operations):
    pop = _population()
    cached = None

    for operation in operations:
        if operation == "step":
            before_t = pop.t.clone()
            pop.step(dt=0.01)
            torch.testing.assert_close(
                pop.t, before_t + torch.as_tensor(0.01, dtype=pop.t.dtype)
            )
        elif operation == "cache":
            pop.cache("sequence")
            cached = (pop.v.detach().clone(), pop.t.detach().clone())
        elif operation == "perturb":
            with torch.no_grad():
                pop.v.add_(3.25)
                pop.t.add_(0.005)
        elif operation == "restore" and cached is not None:
            expected_v, expected_t = cached
            pop.restore("sequence")
            torch.testing.assert_close(
                pop.v, expected_v.to(device=pop.v.device, dtype=pop.v.dtype)
            )
            torch.testing.assert_close(
                pop.t, expected_t.to(device=pop.t.device, dtype=pop.t.dtype)
            )
        elif operation == "initialize":
            pop.initialize()
            torch.testing.assert_close(pop.t, torch.zeros_like(pop.t))
            torch.testing.assert_close(pop.v, torch.full_like(pop.v, -65.0))
        elif operation == "float":
            pop.float()
            assert pop.v.dtype == torch.float32
        elif operation == "double":
            pop.double()
            assert pop.v.dtype == torch.float64

        assert pop.initialized
        assert torch.isfinite(pop.v).all()
        assert torch.isfinite(pop.t).all()
