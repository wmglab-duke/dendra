"""Continuous cache initialization shares native replay and detachment rules."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.networks._continuous_cache import ContinuousCache
from tests.func.test_continuous_con import _clone
from tests.func.test_network import _close
from tests.networks.test_state_cache_translation import (
    CONTINUOUS_WEIGHTS,
    DTYPE,
    OLD_DT,
    _build_continuous_connection,
    _run_continuous_prefix,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
            with torch_compiler_warning_context():
                yield
    finally:
        torch.set_num_threads(previous)


class _RowTransform(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=DTYPE))
        self.register_buffer("offset", torch.tensor(0.05, dtype=DTYPE))

    def forward(self, value):
        # The mean makes full-row replay observably different from applying
        # the transform only to edges whose samples remain pending.
        return value * self.scale + value.mean() * 0.1 + self.offset


def _cache(format="raw"):
    network, source = _build_continuous_connection(OLD_DT, transform=_RowTransform(2.0))
    if format != "future":
        source.enable_state_cache_recording()
    _run_continuous_prefix(network, source)
    cache = source.state_cache()
    if format == "legacy":
        cache["version"] = 2
        cache.pop("value_layout")
        with torch.no_grad():
            cache["pre_value_history"] = torch.stack(
                [source.transform(row) for row in cache["pre_value_history"]]
            )
    return cache


def _target(dt=OLD_DT):
    network, connection = _build_continuous_connection(
        dt,
        weights=CONTINUOUS_WEIGHTS * torch.tensor([1.5, 0.5, -1.25, 0.75], dtype=DTYPE),
        delays=torch.tensor([0.0, 0.2, 0.7, 0.3], dtype=DTYPE),
        transform=_RowTransform(3.0),
    )
    network._reset_runtime_clock(1.35)
    return network, connection


def _parameters(tensors):
    return {
        "weights": tensors.weights,
        "transform_parameters": tensors.transform_parameters,
    }


@pytest.mark.parametrize("format", ("raw", "legacy", "future"))
@pytest.mark.parametrize("new_dt", (0.05, 0.2))
def test_continuous_cache_native_and_pure_restoration_match_current_parameters_and_old_dt(
    format, new_dt
):
    cache = _cache(format)
    network, connection = _target(new_dt)
    native_network, native = _target(new_dt)
    plan, tensors = dn.func.make_functional_continuous_con(connection)
    frozen = _clone((cache, tensors.state))
    normalized = plan.normalize_cache(cache)
    assert isinstance(normalized, ContinuousCache)
    assert all(
        value is None or not value.requires_grad
        for value in pytree.tree_leaves(normalized)
    )
    actual = plan.restore_cache(
        _parameters(tensors), tensors.state, normalized, t=network.t
    )
    if format != "future":
        history = (
            normalized.raw_history
            if format == "raw"
            else normalized.transformed_history
        )
        expected_values = torch.zeros_like(actual["delivery_buffer"])
        expected_mask = torch.zeros_like(actual["delivery_mask"])
        with torch.no_grad():
            for age in range(1, history.shape[0]):
                row = history[age]
                if format == "raw":
                    row = row * 3.0 + row.mean() * 0.1 + 0.05
                for edge in range(plan._connections):
                    remaining = int(plan._delay_steps[edge]) - age
                    if remaining >= 0:
                        target = int(plan._post_idx[edge])
                        expected_values[remaining, target] += (
                            tensors.weights[edge] * row[edge]
                        )
                        expected_mask[remaining, target] = True
        torch.testing.assert_close(actual["delivery_buffer"], expected_values)
        torch.testing.assert_close(actual["delivery_mask"], expected_mask)
    native.initialize_from_state_cache(cache, dt=new_dt, rebuild_delays=False)
    _close(actual, native.state_dict_for_checkpoint())
    assert actual["current_time_step"].item() == 0
    assert not any(value.requires_grad for value in actual.values())
    for index in range(4):
        values = torch.tensor([[0.3 + index * 0.1], [-0.2]], dtype=DTYPE)
        native_network.pre.v = values
        native.syn.reset_continuous_inputs()
        actual, payload, _ = plan.advance(
            tensors.weights,
            actual,
            values,
            transform_parameters=tensors.transform_parameters,
        )
        native.advance()
        torch.testing.assert_close(payload, native.syn.g_pre)
        _close(actual, native.state_dict_for_checkpoint())
    _close((cache, tensors.state), frozen)


@pytest.mark.parametrize("legacy_mask", (False, True))
def test_continuous_future_cache_preserves_zero_presence_and_legacy_inference(
    legacy_mask,
):
    network, connection = _target()
    plan, tensors = dn.func.make_functional_continuous_con(connection)
    values = torch.zeros_like(tensors.state["delivery_buffer"])
    mask = torch.zeros_like(tensors.state["delivery_mask"])
    mask[2, 0] = True
    mask[3, 1] = True
    values[3, 1] = -2.0
    cache = {"dt": OLD_DT, "delivery_buffer": values}
    if not legacy_mask:
        cache["delivery_mask"] = mask
    actual = plan.restore_cache(_parameters(tensors), tensors.state, cache, t=network.t)
    torch.testing.assert_close(actual["delivery_buffer"], values)
    torch.testing.assert_close(
        actual["delivery_mask"], values != 0 if legacy_mask else mask
    )
    assert actual["delivery_mask"][2, 0].item() is not legacy_mask
    assert not torch._C._is_alias_of(actual["delivery_buffer"], values)


def test_continuous_raw_cache_zero_values_rebuild_real_presence_and_detach_all_cache_gradients():
    network, connection = _target()
    plan, tensors = dn.func.make_functional_continuous_con(connection)
    history = torch.zeros(
        (plan._depth, plan._connections), dtype=DTYPE, requires_grad=True
    )
    parameters = _parameters(tensors)
    parameters["weights"] = torch.zeros_like(tensors.weights, requires_grad=True)
    parameters["transform_parameters"] = {
        "scale": tensors.transform_parameters["scale"],
        "offset": torch.zeros_like(
            tensors.transform_parameters["offset"], requires_grad=True
        ),
    }
    cache = plan.normalize_cache(
        {"dt": OLD_DT, "version": 4, "pre_value_history": history}
    )
    actual = plan.restore_cache(parameters, tensors.state, cache, t=network.t)
    assert not actual["delivery_buffer"].any()
    assert actual["delivery_mask"].any()
    assert not any(value.requires_grad for value in actual.values())
    derivative = torch.func.grad(
        lambda weights: plan.restore_cache(
            {**parameters, "weights": weights}, tensors.state, cache, t=network.t
        )["delivery_buffer"].sum()
    )(parameters["weights"])
    torch.testing.assert_close(derivative, torch.zeros_like(derivative))
    following, payload, _ = plan.advance(
        tensors.weights,
        actual,
        torch.ones(plan.source_shape, dtype=DTYPE),
        transform_parameters=tensors.transform_parameters,
    )
    gradients = torch.autograd.grad(
        following["delivery_buffer"].sum() + payload.sum(),
        (tensors.weights, tensors.transform_parameters["scale"]),
    )
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )


def test_continuous_normalized_cache_restoration_compiles_without_native_mutation():
    network, connection = _target()
    plan, tensors = dn.func.make_functional_continuous_con(connection)
    cache = plan.normalize_cache(_cache())
    frozen = _clone((cache, tensors.state))
    expected = plan.restore_cache(
        _parameters(tensors), tensors.state, cache, t=network.t
    )
    execute = torch.compile(plan.restore_cache, backend="aot_eager", fullgraph=True)
    actual = execute(_parameters(tensors), tensors.state, cache, t=network.t)
    _close(actual, expected)
    _close((cache, tensors.state), frozen)
    _close(connection.state_dict_for_checkpoint(), tensors.state)


@pytest.mark.parametrize("lanes", (0, 2))
def test_continuous_normalized_cache_restoration_maps_weights_and_start_times(lanes):
    network, connection = _target()
    plan, tensors = dn.func.make_functional_continuous_con(connection)
    cache = plan.normalize_cache(_cache())
    weights = tensors.weights.expand(lanes, -1).clone()
    times = torch.linspace(0.25, 0.45, lanes, dtype=DTYPE)
    actual = torch.vmap(
        lambda weight, time: plan.restore_cache(
            {"weights": weight, "transform_parameters": tensors.transform_parameters},
            tensors.state,
            cache,
            t=time,
        )
    )(weights, times)
    assert actual["delivery_buffer"].shape == (lanes, plan._depth, plan._width)
    for index in range(lanes):
        expected = plan.restore_cache(
            {
                "weights": weights[index],
                "transform_parameters": tensors.transform_parameters,
            },
            tensors.state,
            cache,
            t=times[index],
        )
        _close(pytree.tree_map(lambda value: value[index], actual), expected)


@pytest.mark.parametrize("lanes", (0, 2))
@pytest.mark.parametrize("mapped", ("history", "transform"))
def test_continuous_cache_maps_history_and_reducing_transform_parameters(lanes, mapped):
    network, connection = _target()
    plan, tensors = dn.func.make_functional_continuous_con(connection)
    cache = plan.normalize_cache(_cache())
    parameters = _parameters(tensors)
    samples = (
        cache.raw_history.expand(lanes, -1, -1).clone()
        if mapped == "history"
        else tensors.transform_parameters["scale"].expand(lanes).clone()
    )

    def restore(value):
        current_cache = cache
        current_parameters = parameters
        if mapped == "history":
            current_cache = cache._replace(raw_history=value)
        else:
            current_parameters = {
                **parameters,
                "transform_parameters": {
                    **parameters["transform_parameters"],
                    "scale": value,
                },
            }
        return plan.restore_cache(
            current_parameters, tensors.state, current_cache, t=network.t
        )

    actual = torch.vmap(restore)(samples)
    assert actual["delivery_buffer"].shape == (lanes, plan._depth, plan._width)
    for lane in range(lanes):
        _close(
            pytree.tree_map(lambda value: value[lane], actual), restore(samples[lane])
        )


@pytest.mark.parametrize("invalid", ("history_width", "mask_shape", "dt", "layout"))
def test_continuous_cache_rejects_invalid_data_without_mutating_inputs(invalid):
    network, connection = _target()
    plan, tensors = dn.func.make_functional_continuous_con(connection)
    cache = _cache()
    if invalid == "history_width":
        cache["pre_value_history"] = cache["pre_value_history"][:, :1]
    elif invalid == "mask_shape":
        cache = {
            "delivery_buffer": torch.zeros_like(tensors.state["delivery_buffer"]),
            "delivery_mask": torch.zeros(1, 1),
        }
    elif invalid == "dt":
        cache["dt"] = 0.0
    else:
        values = torch.zeros((plan._depth, plan._connections), dtype=DTYPE)
        cache = ContinuousCache(values, values, None, None)
    frozen = _clone((cache, tensors.state))
    with pytest.raises((ValueError, TypeError)):
        plan.restore_cache(_parameters(tensors), tensors.state, cache, t=network.t)
    _close((cache, tensors.state), frozen)


@pytest.mark.parametrize(
    "hook",
    ("initialize_from_state_cache", "_materialize_delivery_from_pre_value_history"),
)
def test_continuous_cache_restoration_rejects_overridden_native_cache_hooks(hook):
    network, connection = _target()
    plan, tensors = dn.func.make_functional_continuous_con(connection)
    cache = plan.normalize_cache(_cache())
    frozen = _clone(tensors.state)
    setattr(connection, hook, lambda *args, **kwargs: None)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical"):
        plan.restore_cache(_parameters(tensors), tensors.state, cache, t=network.t)
    _close(tensors.state, frozen)
