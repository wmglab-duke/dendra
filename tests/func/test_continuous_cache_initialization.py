"""Continuous cache initialization shares native replay and detachment rules."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import graded_syn, pas
from dendra.models.networks._continuous_cache import (
    ContinuousCache,
    continuous_cache_routes,
    continuous_history_delivery,
)
from dendra.models.networks._continuous_delivery import (
    continuous_clear,
    continuous_enqueue_linear,
)
from dendra.models.parametric import Bounded
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


def test_fractional_cache_replays_current_delays_through_upper_tap_capacity():
    cap = 4.75 * OLD_DT
    delays = OLD_DT * torch.tensor([2.0, 4.5, 1.5, 3.0], dtype=DTYPE)
    network, native = _build_continuous_connection(
        OLD_DT, delays=delays, max_delay_ms=cap, transform=_RowTransform(3.0)
    )
    native.set_diff_config(diff_delays=True)
    native.enable_state_cache_recording()
    _run_continuous_prefix(network, native)
    cache = native.state_cache()
    slot = native.current_time_step.item()
    expected = native.delivery_buffer.roll(-slot, dims=0)
    expected_mask = native.delivery_mask.roll(-slot, dims=0)
    assert expected_mask[-2].any()

    _, connection = _build_continuous_connection(
        OLD_DT,
        delays=torch.full_like(delays, 2.5 * OLD_DT),
        max_delay_ms=cap,
        transform=_RowTransform(3.0),
    )
    connection.set_diff_config(diff_delays=True)
    plan, tensors = dn.func.make_functional_continuous_con(connection)
    restore = torch.compile(plan.restore_cache, backend="aot_eager", fullgraph=True)
    actual = restore(
        _parameters(tensors),
        tensors.state,
        plan.normalize_cache(cache),
        t=network.t,
        delays=delays.requires_grad_(),
    )
    native.initialize_from_state_cache(cache)
    _close(actual, native.state_dict_for_checkpoint())
    torch.testing.assert_close(actual["delivery_buffer"], expected)
    torch.testing.assert_close(actual["delivery_mask"], expected_mask)
    assert not any(value.requires_grad for value in actual.values())


@pytest.mark.parametrize(
    ("lower_offsets", "samples", "pending_row"),
    (
        ((1, 1, 1), ((1e8, 0.0, 0.0), (-1e8, 2.0, 0.0)), 0),
        ((2, 1, 2), ((0.0, 0.0, 0.0), (1e8, 2.0, -1e8)), 1),
    ),
)
def test_fractional_cache_preserves_live_accumulation_order(
    lower_offsets, samples, pending_row
):
    depth = 4
    samples = torch.tensor(samples, dtype=torch.float32)
    offsets = torch.tensor(lower_offsets)
    post_idx = torch.zeros_like(offsets)
    fraction = torch.full_like(samples[0], 0.5)
    queue = samples.new_zeros((depth, 1))
    mask = torch.zeros_like(queue, dtype=torch.bool)
    for index, sample in enumerate(samples):
        slot = torch.tensor([index])
        queue, mask = continuous_clear(queue, mask, slot)
        queue, mask = continuous_enqueue_linear(
            queue,
            mask,
            slot,
            sample,
            post_idx=post_idx,
            delay_steps=offsets,
            fraction=fraction,
            depth=depth,
            width=1,
        )

    routes = continuous_cache_routes(
        offsets, post_idx, depth=depth, width=1, fraction=fraction
    )
    replayed, presence = continuous_history_delivery(
        torch.cat((torch.zeros_like(samples[:1]), samples.flip(0))),
        torch.ones_like(fraction),
        depth=depth,
        width=1,
        routes=routes,
    )
    assert replayed[pending_row, 0].item() == 1.0
    torch.testing.assert_close(replayed, queue.roll(-len(samples), dims=0))
    torch.testing.assert_close(presence, mask.roll(-len(samples), dims=0))


def test_network_initialization_applies_prebuild_interpolation_to_warm_cache():
    cap = 4.75 * OLD_DT
    weight = 0.01

    def make_network(delay):
        pre = dn.SingleCompartment(N=1, C=1)
        post = dn.SingleCompartment(N=1, C=1)
        for population in (pre, post):
            population.insert(pas, g=0.001, e=-65.0)
        post.insert(graded_syn, e=-10.0)
        network = dn.Network({"pre": pre, "post": post})
        network.connect_continuous_one_to_one(
            pre[:],
            post[:],
            post.mech.graded_syn,
            input="g_pre",
            weight=weight,
            delay=delay,
        )
        return network.eval()

    warm = make_network(OLD_DT)
    warm.initialize(OLD_DT, max_delay_ms=cap)
    warm._enable_synapse_state_cache_recording()
    with torch.no_grad():
        for value in (0.1, 0.2, 0.3, 0.4):
            warm.pre.v.fill_(value)
            warm.step()
    warm.cache_state()

    restored = make_network(2.5 * OLD_DT)
    restored.load_state_cache(warm._state_cache, warm._syn_cache)
    restored.set_synaptic_diff_config(kind="continuous", diff_delays=True)
    restored.initialize(OLD_DT, max_delay_ms=cap)
    connection = next(iter(restored.continuous_synapses.values()))
    # Half of each sample is due after two steps and the other half after three.
    expected = weight * torch.tensor([0.25, 0.35, 0.2, 0.0, 0.0, 0.0], dtype=DTYPE)
    torch.testing.assert_close(connection.delivery_buffer[:, 0], expected)
    assert connection.delivery_mask[:, 0].nonzero().flatten().tolist() == [0, 1, 2]


@pytest.mark.parametrize("horizon", (None, 4))
def test_recorded_history_keeps_its_ages_when_interpolation_resizes_delivery(horizon):
    delay = Bounded(0.0, requires_grad=False)
    network, connection = _build_continuous_connection(
        OLD_DT, delays=delay, max_delay_ms=4.75 * OLD_DT
    )
    connection.global_step.fill_(-3)
    connection.enable_state_cache_recording(horizon_steps=horizon)
    with torch.no_grad():
        for value in (1, 2):
            network.pre.v.fill_(value)
            connection.advance()
        delay.set(2.5 * OLD_DT)
        connection.delay_ms.init(reinit=True)
        connection.set_diff_config(diff_delays=True)
        history = connection.state_cache()["pre_value_history"]
        expected_horizon = connection.max_delay_steps if horizon is None else horizon
        assert history.shape[0] == expected_horizon
        torch.testing.assert_close(history[1:3, 0], history.new_tensor([2.0, 1.0]))

        for value in range(3, 11):
            network.pre.v.fill_(value)
            connection.syn.reset_continuous_inputs()
            connection.advance()
        slot = connection.current_time_step.item()
        expected = connection.delivery_buffer.roll(-slot, dims=0)
        cache = connection.state_cache()
        connection.initialize_from_state_cache(cache)
    torch.testing.assert_close(connection.delivery_buffer, expected)


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
    (
        "initialize_from_state_cache",
        "_materialize_delivery_from_pre_value_history",
        "_continuous_cache_routes",
    ),
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
