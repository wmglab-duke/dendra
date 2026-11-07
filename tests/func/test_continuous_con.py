"""Continuous projections preserve analog phases, masks, and explicit gradients."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import graded_syn, sigmoid_release

DT = 0.1
DTYPE = torch.float64
pytestmark = pytest.mark.cpu


def _connection(delays=(0.0, 0.1, 0.0, 0.2), *, transform=None, batch=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        pre = dn.SingleCompartment(N=2, C=1, dtype=DTYPE)
        post = dn.SingleCompartment(N=2, C=1, dtype=DTYPE)
        post.insert(graded_syn)
        net = dn.Network({"pre": pre, "post": post})
        net.connect_continuous(
            pre[:],
            post[:],
            post.mech.graded_syn,
            conn_spec="all_to_all",
            weight=torch.tensor([1.0, -2.0, 0.0, 0.5], dtype=DTYPE),
            delay=torch.tensor(delays, dtype=DTYPE),
            transform=transform,
        )
        if batch is not None:
            net.batch(batch)
        net.train()
        net.initialize(DT)
    return net, next(iter(net.continuous_synapses.values()))


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected):
    left, left_spec = pytree.tree_flatten(actual)
    right, right_spec = pytree.tree_flatten(expected)
    assert left_spec == right_spec
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize(
    "delays",
    (
        (0.0, 0.0, 0.0, 0.0),
        (0.1, 0.1, 0.1, 0.1),
        (0.1, 0.2, 0.1, 0.2),
        (0.0, 0.2, 0.0, 0.2),
        (0.0, 0.1, 0.0, 0.2),
    ),
)
@pytest.mark.parametrize("batch", (None, 2))
def test_native_phase_payload_mask_carry_and_gradients(delays, batch):
    net, native = _connection(delays, batch=batch)
    projection, tensors = dn.func.make_functional_continuous_con(native)
    original = _clone(tensors.state)
    weights = tensors.weights.detach().requires_grad_()
    state = tensors.state
    actual, expected = [], []
    tape = (
        torch.arange(6 * net.pre.v.numel(), dtype=DTYPE)
        .reshape(6, *net.pre.v.shape)
        .mul(0.1)
        .sub(0.4)
        .requires_grad_()
    )
    for values in tape:
        net.pre.v = values
        native.syn.reset_continuous_inputs()
        if projection.receive_before_source:
            read_state, before, before_mask = projection.read_due(state)
            _close(read_state, state)
        state, payload, mask = projection.advance(weights, state, values)
        if projection.receive_before_source:
            torch.testing.assert_close(payload, before)
            torch.testing.assert_close(mask, before_mask)
        native.advance()
        torch.testing.assert_close(payload, native.syn.g_pre)
        _close(state, native.state_dict_for_checkpoint())
        actual.append(payload)
        expected.append(native.syn.g_pre)
        # Presence is independent of zero-valued and zero-weight traffic.
        assert mask.dtype is torch.bool
    functional_grads = torch.autograd.grad(torch.stack(actual).sum(), (weights, tape))
    native_grads = torch.autograd.grad(
        torch.stack(expected).sum(), (native.weight.w, tape)
    )
    _close(functional_grads, native_grads)
    _close(tensors.state, original)


def test_extraction_checkpoint_parameter_snapshots_and_fresh_continuation():
    _, native = _connection()
    projection, tensors = dn.func.make_functional_continuous_con(native)
    source_before = _clone(native.state_dict_for_checkpoint())
    values = torch.tensor([2.0, -1.0], dtype=DTYPE).reshape(projection.source_shape)
    weights = tensors.weights.detach().requires_grad_()
    state, _, _ = projection.advance(weights, tensors.state, values)
    saved = pytree.tree_map_only(torch.Tensor, torch.clone, state)
    suffix, payload, mask = projection.advance(weights * 3, saved, values * 0)
    assert bool(mask.any())
    gradients = torch.autograd.grad(payload.sum(), weights)[0]
    assert bool((gradients != 0).any())
    _close(native.state_dict_for_checkpoint(), source_before)
    _, fresh = _connection()
    assert projection.commit_state_(fresh, saved) is fresh
    _close(projection.extract(fresh).state, saved)
    fresh.pre.v = values * 0
    fresh.syn.reset_continuous_inputs()
    fresh.advance()
    _close(fresh.state_dict_for_checkpoint(), suffix)
    torch.testing.assert_close(fresh.syn.g_pre, payload)
    for name, value in saved.items():
        assert fresh._buffers[name] is not value


@pytest.mark.parametrize(
    "transform",
    (
        None,
        torch.sigmoid,
        torch.tanh,
        torch.relu,
        torch.nn.Identity(),
        torch.nn.Sigmoid(),
        torch.nn.Tanh(),
        torch.nn.ReLU(),
        sigmoid_release(theta=-0.2, sigma=0.5),
    ),
)
def test_parameter_free_transforms_match_native(transform):
    _, native = _connection((0.0,) * 4, transform=transform)
    projection, tensors = dn.func.make_functional_continuous_con(native)
    values = torch.tensor([-1.0, 0.5], dtype=DTYPE).reshape(projection.source_shape)
    native.pre.v = values
    native.syn.reset_continuous_inputs()
    native.advance()
    _, payload, mask = projection.advance(tensors.weights, tensors.state, values)
    torch.testing.assert_close(payload, native.syn.g_pre)
    assert bool(mask.all())


@pytest.mark.parametrize("backend", (None, "aot_eager", "inductor"))
def test_release_explicit_parameters_source_and_weight_gradients(backend):
    _, native = _connection(
        (0.0,) * 4, transform=sigmoid_release(theta=-0.2, sigma=0.5)
    )
    # Ordinary nn.Module registration makes these explicit values trainable;
    # float activation metadata remains frozen for the default module.
    native.transform.theta = torch.nn.Parameter(torch.tensor(-0.2, dtype=DTYPE))
    native.transform.sigma = torch.nn.Parameter(torch.tensor(0.5, dtype=DTYPE))
    projection, tensors = dn.func.make_functional_continuous_con(native)
    theta = tensors.transform_parameters["theta"].requires_grad_()
    sigma = tensors.transform_parameters["sigma"].requires_grad_()
    weights = tensors.weights.detach().requires_grad_()
    values = (
        torch.tensor([-0.5, 0.8], dtype=DTYPE)
        .reshape(projection.source_shape)
        .requires_grad_()
    )

    def evaluate(weight, source, theta, sigma):
        return projection.advance(
            weight,
            tensors.state,
            source,
            transform_parameters={"theta": theta, "sigma": sigma},
        )

    run = (
        evaluate
        if backend is None
        else torch.compile(evaluate, backend=backend, fullgraph=True)
    )
    _, payload, mask = run(weights, values, theta, sigma)
    expected_values = torch.sigmoid(
        (values.reshape(-1)[native.pre_idx] - theta) / sigma
    )
    expected = torch.zeros(2, dtype=DTYPE).index_add(
        0, native.post_idx, weights * expected_values
    )
    torch.testing.assert_close(payload.reshape(-1), expected)
    torch.testing.assert_close(mask, torch.ones_like(mask))
    actual_grad = torch.autograd.grad(
        payload.sum(), (weights, values, theta, sigma), retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected.sum(), (weights, values, theta, sigma))
    _close(actual_grad, expected_grad)
    # Parameter updates remain explicit and cannot mutate the native transform.
    _, changed, _ = run(weights, values, theta + 0.4, sigma * 2)
    assert not torch.equal(changed, payload)
    assert native.transform.theta == -0.2
    assert native.transform.sigma == 0.5


@pytest.mark.parametrize("lanes", (0, 2))
@pytest.mark.parametrize("mapped", ("weights", "source", "state", "slot", "all"))
def test_vmap_independent_inputs_and_empty_lanes(lanes, mapped):
    _, native = _connection()
    projection, tensors = dn.func.make_functional_continuous_con(native)
    values = torch.tensor([-1.0, 2.0], dtype=DTYPE).reshape(projection.source_shape)
    arguments = (tensors.weights, tensors.state, values)
    dims = (
        0 if mapped in ("weights", "all") else None,
        0 if mapped in ("state", "all") else None,
        0 if mapped in ("source", "all") else None,
    )
    if mapped == "slot":
        states = dict(tensors.state)
        states["current_time_step"] = (
            states["current_time_step"].expand(lanes, 1).clone()
        )
        dimensions = {
            name: (0 if name == "current_time_step" else None) for name in states
        }
        actual = torch.vmap(
            lambda carry: projection.advance(tensors.weights, carry, values),
            in_dims=(dimensions,),
        )(states)
    else:
        inputs = tuple(
            pytree.tree_map_only(
                torch.Tensor, lambda x: x.expand(lanes, *x.shape).clone(), value
            )
            if dim == 0
            else value
            for value, dim in zip(arguments, dims, strict=True)
        )
        actual = torch.vmap(projection.advance, in_dims=dims)(*inputs)
    if lanes:
        expected = projection.advance(*arguments)
        for value, reference in zip(
            pytree.tree_leaves(actual), pytree.tree_leaves(expected), strict=True
        ):
            torch.testing.assert_close(value, reference.expand(lanes, *reference.shape))
    else:
        assert all(value.shape[0] == 0 for value in pytree.tree_leaves(actual))


def test_forward_reverse_jacobians_and_pending_queue_graph():
    _, native = _connection()
    projection, tensors = dn.func.make_functional_continuous_con(native)
    values = torch.tensor([1.0, -2.0], dtype=DTYPE).reshape(projection.source_shape)

    def evaluate(weights, queue):
        state = {**tensors.state, "delivery_buffer": queue}
        next_state, payload, _ = projection.advance(weights, state, values)
        return next_state["delivery_buffer"].sum() + payload.sum()

    args = (tensors.weights, tensors.state["delivery_buffer"])
    _close(
        torch.func.jacfwd(evaluate, argnums=(0, 1))(*args),
        torch.func.jacrev(evaluate, argnums=(0, 1))(*args),
    )


@pytest.mark.parametrize("delays", ((0.1, 0.2, 0.1, 0.2), (0.0, 0.1, 0.0, 0.2)))
@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_compiled_delay_carry_and_pending_gradients(delays, backend):
    _, native = _connection(delays)
    projection, tensors = dn.func.make_functional_continuous_con(native)
    values = torch.tensor([1.0, -2.0], dtype=DTYPE).reshape(projection.source_shape)
    state = {
        **tensors.state,
        "delivery_buffer": tensors.state["delivery_buffer"].clone().requires_grad_(),
    }
    weights = tensors.weights.detach().requires_grad_()

    def evaluate(weights, state, values):
        total = values.new_zeros(())
        for _ in range(3):
            state, payload, _mask = projection.advance(weights, state, values)
            total = total + payload.sum()
        return state, total + state["delivery_buffer"].sum()

    expected = evaluate(weights, state, values)
    actual = torch.compile(evaluate, backend=backend, fullgraph=True)(
        weights, state, values
    )
    _close(actual, expected)
    _close(
        torch.autograd.grad(actual[1], (weights, state["delivery_buffer"])),
        torch.autograd.grad(expected[1], (weights, state["delivery_buffer"])),
    )


@pytest.mark.parametrize("mutation", ("routing", "delays", "timestep", "transform"))
def test_cold_extraction_and_commit_reject_incompatible_metadata(mutation):
    _, native = _connection()
    projection, tensors = dn.func.make_functional_continuous_con(native)
    _, target = _connection()
    if mutation == "routing":
        target.post_idx = target.post_idx.roll(1)
    elif mutation == "delays":
        target.delay_steps = target.delay_steps + 1
    elif mutation == "timestep":
        target.dt = target.dt * 2
    else:
        target.transform = torch.nn.Sigmoid()
    before = _clone(target.state_dict_for_checkpoint())
    with pytest.raises(dn.func.FunctionalizationError, match="does not match"):
        projection.extract(target)
    with pytest.raises(dn.func.FunctionalizationError, match="does not match"):
        projection.commit_state_(target, tensors.state)
    _close(target.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("hook", ("_pre_value", "get_pre_var"))
def test_lowering_rejects_custom_source_callbacks_without_invoking_them(hook):
    _, native = _connection()
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("must not evaluate custom source")

    setattr(native, hook, forbidden)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical"):
        dn.func.make_functional_continuous_con(native)
    assert not calls


def test_empty_routing_advances_counters_without_receiving_or_clearing_pending_rows():
    _, native = _connection()
    # Public builders omit empty projections. The canonical internal object can
    # still retain an empty routing table and a preallocated ring.
    native.pre_idx = native.pre_idx[:0]
    native.post_idx = native.post_idx[:0]
    native.delay_steps = native.delay_steps[:0]
    native.weight.w = native.weight.w[:0]
    native.delay_ms.w = native.delay_ms.w[:0]
    native._rebuild_delay_metadata()
    native._configure_advance_impl()
    native.current_time_step.fill_(1)
    native.delivery_buffer.fill_(3)
    native.delivery_mask.fill_(True)
    projection, tensors = dn.func.make_functional_continuous_con(native)
    assert not projection.has_connections
    state, payload, mask = projection.advance(
        tensors.weights,
        tensors.state,
        torch.zeros(projection.source_shape, dtype=DTYPE),
    )
    native.advance()
    _close(state, native.state_dict_for_checkpoint())
    torch.testing.assert_close(payload, torch.zeros_like(payload))
    assert not bool(mask.any())


@pytest.mark.parametrize(
    "mutation",
    (
        "initialize",
        "dtype",
        "input",
        "reduce",
        "transform",
        "advance",
        "getter",
        "helper",
    ),
)
def test_source_configuration_changes_reject_calls_without_mutating_carry(mutation):
    _, native = _connection()
    projection, tensors = dn.func.make_functional_continuous_con(native)
    before = _clone(tensors.state)
    if mutation == "initialize":
        native.initialize(reinit_delays=False)
    elif mutation == "dtype":
        native.float()
    elif mutation == "input":
        native.input = "other"
    elif mutation == "reduce":
        native.reduce = "max"
    elif mutation == "transform":
        native.transform = torch.nn.Sigmoid()
    elif mutation == "advance":
        native.advance = lambda: None
    elif mutation == "getter":
        native.get_pre_var = lambda source: source.v
    else:
        native._pre_value = lambda: torch.zeros(4, dtype=DTYPE)
    with pytest.raises(dn.func.FunctionalizationError, match="changed"):
        projection.read_due(tensors.state)
    _close(tensors.state, before)


@pytest.mark.parametrize(
    "transform",
    (
        torch.nn.Dropout(),
        torch.nn.ReLU(inplace=True),
    ),
)
def test_impure_transforms_reject_without_source_mutation(transform):
    _, native = _connection(transform=transform)
    before = _clone(native.state_dict_for_checkpoint())
    with (
        dn.ctx(CHECK_PURITY=True),
        pytest.raises(dn.func.FunctionalizationError, match="transform"),
    ):
        dn.func.make_functional_continuous_con(native)
    _close(native.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize(
    "hook", ("_call_impl", "_compiled_call_impl", "_wrapped_call_impl")
)
@pytest.mark.parametrize("lowered", (False, True))
def test_transform_custom_call_dispatch_rejects_without_execution(hook, lowered):
    _, native = _connection(transform=sigmoid_release())
    if lowered:
        projection, tensors = dn.func.make_functional_continuous_con(native)
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("must not execute custom dispatch")

    setattr(native.transform, hook, forbidden)
    with pytest.raises(dn.func.FunctionalizationError, match="transform|canonical"):
        if lowered:
            projection.read_due(tensors.state)
        else:
            dn.func.make_functional_continuous_con(native)
    assert not calls


@pytest.mark.parametrize("field", ("weights", "source", "state", "transform"))
def test_invalid_explicit_inputs_reject_without_mutation(field):
    _, native = _connection(transform=sigmoid_release())
    projection, tensors = dn.func.make_functional_continuous_con(native)
    weights, state = tensors.weights, tensors.state
    values = torch.zeros(projection.source_shape, dtype=DTYPE)
    parameters = tensors.transform_parameters
    before = _clone(state)
    if field == "weights":
        weights = weights.float()
    elif field == "source":
        values = values.reshape(-1)
    elif field == "state":
        state = {**state, "extra": values}
    else:
        parameters = {"extra": values}
    with pytest.raises((TypeError, ValueError)):
        projection.advance(weights, state, values, transform_parameters=parameters)
    _close(tensors.state, before)
