"""Continuous transforms execute their own pure forward with explicit tensors."""

from __future__ import annotations

import random
import sys
from types import MappingProxyType, ModuleType

import numpy as np
import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._continuous_transform import ContinuousTransformPlan
from tests.func.test_continuous_con import DTYPE, _clone, _close, _connection

pytestmark = pytest.mark.cpu


def _signed_square(value):
    return value * value.abs()


def _nested_transform(value):
    def soften(selected):
        return torch.tanh(selected * 0.7)

    return _signed_square(soften(value)) + 0.15


def _immutable_closure():
    scale, offsets = 1.3, (0.1, -0.2)

    def transform(value):
        return torch.sin(value * scale) + offsets[0] + offsets[1]

    return transform


class _AffineGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.2, dtype=DTYPE))
        self.bias = torch.nn.Parameter(torch.tensor(-0.15, dtype=DTYPE))
        self.register_buffer("offset", torch.tensor(0.07, dtype=DTYPE))
        self.gain = 0.8

    def forward(self, value):
        return torch.tanh(value * self.scale + self.bias + self.offset) * self.gain


class _TiedGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor(0.4, dtype=DTYPE))
        self.right = self.left

    def forward(self, value):
        return value * self.left + torch.sin(value) * self.right


class _BroadcastAffineGate(_AffineGate):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(self.scale.detach().reshape(1))
        self.bias = torch.nn.Parameter(self.bias.detach().reshape(1))
        self.offset = self.offset.reshape(1)


@pytest.mark.parametrize("factory", (lambda: torch.sin, _AffineGate))
def test_transform_plan_cpu_device_alias_matches_real_tensor_storage(factory):
    transform = factory()
    plan = ContinuousTransformPlan(transform, size=4, device="cpu:0", dtype=DTYPE)
    parameters = plan.extract(transform)
    values = torch.linspace(-0.8, 0.8, 4, dtype=DTYPE)
    assert plan.device == values.device
    torch.testing.assert_close(plan.apply(values, parameters), transform(values))


@pytest.mark.parametrize(
    "transform",
    (
        torch.sin,
        torch.square,
        _signed_square,
        _nested_transform,
        _immutable_closure(),
        lambda value: torch.cos(value) + value.square() / 3,
    ),
)
def test_plain_functions_execute_the_same_native_callable(transform):
    _, native = _connection((0.0,) * 4, transform=transform)
    projection, tensors = dn.func.make_functional_continuous_con(native)
    assert tensors.transform_parameters is None
    values = (
        torch.tensor([-0.8, 0.6], dtype=DTYPE)
        .reshape(projection.source_shape)
        .requires_grad_()
    )
    native.pre.v = values
    native.syn.reset_continuous_inputs()
    native.advance()
    _, payload, mask = projection.advance(tensors.weights, tensors.state, values)
    torch.testing.assert_close(payload, native.syn.g_pre)
    assert mask.all()
    _close(
        torch.autograd.grad(payload.square().sum(), (tensors.weights, values)),
        torch.autograd.grad(native.syn.g_pre.square().sum(), (native.weight.w, values)),
    )


def test_pure_transform_with_restricted_domain_is_not_rejected_for_probe_nans():
    _, native = _connection((0.0,) * 4, transform=torch.log)
    projection, tensors = dn.func.make_functional_continuous_con(native)
    values = torch.tensor([[0.4], [1.7]], dtype=DTYPE)
    native.pre.v = values
    native.syn.reset_continuous_inputs()
    native.advance()
    _, payload, _ = projection.advance(tensors.weights, tensors.state, values)
    torch.testing.assert_close(payload, native.syn.g_pre)


@pytest.mark.parametrize(
    "factory",
    (
        _AffineGate,
        lambda: torch.nn.Linear(4, 4, dtype=DTYPE),
        lambda: torch.nn.PReLU(num_parameters=1).to(dtype=DTYPE),
        lambda: torch.nn.Sequential(
            torch.nn.Linear(4, 4, dtype=DTYPE), torch.nn.Tanh()
        ),
    ),
)
def test_registered_modules_nested_parameters_native_values_and_gradients(factory):
    _, native = _connection(transform=factory())
    for value in native.transform.buffers():
        value.requires_grad_()
    projection, tensors = dn.func.make_functional_continuous_con(native)
    native_parameters = dict(native.transform.named_parameters())
    native_parameters.update(dict(native.transform.named_buffers()))
    assert set(tensors.transform_parameters) == set(native_parameters)
    for name, value in tensors.transform_parameters.items():
        assert value.is_leaf
        assert value.requires_grad == native_parameters[name].requires_grad
        assert value.data_ptr() != native_parameters[name].data_ptr()
    original = _clone(native.transform.state_dict())
    weights = tensors.weights.detach().clone().requires_grad_()
    native_weights = weights.detach().clone().requires_grad_()
    native.weight.w = native_weights
    tape = torch.tensor(
        [[[-0.8], [0.6]], [[0.3], [-0.4]], [[-0.2], [0.7]], [[0.5], [0.1]]],
        dtype=DTYPE,
        requires_grad=True,
    )
    native_tape = tape.detach().clone().requires_grad_()
    pending = (tensors.state["delivery_buffer"] + 0.03).requires_grad_()
    native_pending = pending.detach().clone().requires_grad_()
    native.delivery_buffer = native_pending
    presence = torch.ones_like(tensors.state["delivery_mask"])
    native.delivery_mask = presence.clone()
    state = {**tensors.state, "delivery_buffer": pending, "delivery_mask": presence}
    loss, native_loss = 0, 0
    for step, (values, reference_values) in enumerate(zip(tape, native_tape)):
        state, payload, _ = projection.advance(
            weights,
            state,
            values,
            transform_parameters=tensors.transform_parameters,
        )
        native.pre.v = reference_values
        native.syn.reset_continuous_inputs()
        native.advance()
        torch.testing.assert_close(payload, native.syn.g_pre)
        _close(state, native.state_dict_for_checkpoint())
        loss = (
            loss
            + (step + 1) * payload.square().sum()
            + state["delivery_buffer"].square().sum()
        )
        native_loss = (
            native_loss
            + (step + 1) * native.syn.g_pre.square().sum()
            + native.delivery_buffer.square().sum()
        )
    actual_inputs = (weights, tape, pending, *tensors.transform_parameters.values())
    native_inputs = (
        native_weights,
        native_tape,
        native_pending,
        *(native_parameters[name] for name in tensors.transform_parameters),
    )
    actual = torch.autograd.grad(loss, actual_inputs)
    expected = torch.autograd.grad(native_loss, native_inputs)
    _close(actual, expected)
    assert all(torch.count_nonzero(value) for value in actual)
    _close(native.transform.state_dict(), original)


def test_tied_registered_parameter_aliases_keep_the_native_gradient():
    _, native = _connection((0.0,) * 4, transform=_TiedGate())
    projection, tensors = dn.func.make_functional_continuous_con(native)
    parameters = tensors.transform_parameters
    # Extraction must preserve one explicit tensor identity for a native tie.
    assert "left" in parameters
    if "right" in parameters:
        assert parameters["right"] is parameters["left"]
    values = torch.tensor([[-0.8], [0.6]], dtype=DTYPE)
    _, payload, _ = projection.advance(
        tensors.weights, tensors.state, values, transform_parameters=parameters
    )
    native.pre.v = values
    native.syn.reset_continuous_inputs()
    native.advance()
    torch.testing.assert_close(payload, native.syn.g_pre)
    actual = torch.autograd.grad(payload.sum(), parameters["left"])[0]
    expected = torch.autograd.grad(native.syn.g_pre.sum(), native.transform.left)[0]
    torch.testing.assert_close(actual, expected)


def test_read_only_transform_packet_preserves_tied_native_parameter_gradients():
    _, native = _connection((0.0,) * 4, transform=_TiedGate())
    projection, tensors = dn.func.make_functional_continuous_con(native)
    parameters = MappingProxyType(tensors.transform_parameters)
    if "right" in parameters:
        assert parameters["right"] is parameters["left"]
    values = torch.tensor([[-0.8], [0.6]], dtype=DTYPE, requires_grad=True)
    native_values = values.detach().clone().requires_grad_()
    _, payload, _ = projection.advance(
        tensors.weights, tensors.state, values, transform_parameters=parameters
    )
    native.pre.v = native_values
    native.syn.reset_continuous_inputs()
    native.advance()
    torch.testing.assert_close(payload, native.syn.g_pre)
    actual = torch.autograd.grad(
        payload.square().sum(), (parameters["left"], values, tensors.weights)
    )
    expected = torch.autograd.grad(
        native.syn.g_pre.square().sum(),
        (native.transform.left, native_values, native.weight.w),
    )
    _close(actual, expected)
    assert all(value.abs().sum() > 0 for value in actual)
    assert parameters["left"] is tensors.transform_parameters["left"]


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_compiled_generic_module_parameters_buffers_and_pending_gradients(backend):
    _, native = _connection(transform=_AffineGate())
    projection, tensors = dn.func.make_functional_continuous_con(native)
    weights = tensors.weights.detach().requires_grad_()
    values = torch.tensor([[-0.8], [0.6]], dtype=DTYPE, requires_grad=True)
    parameters = {
        name: value.detach().clone().requires_grad_()
        for name, value in tensors.transform_parameters.items()
    }
    pending = (tensors.state["delivery_buffer"] + 0.04).requires_grad_()

    def evaluate(weight, source, parameters, queue):
        state = {**tensors.state, "delivery_buffer": queue}
        loss = source.new_zeros(())
        for scale in (1.0, 0.3, -0.2):
            state, payload, _ = projection.advance(
                weight, state, source * scale, transform_parameters=parameters
            )
            loss = loss + payload.square().sum()
        return state, loss + state["delivery_buffer"].square().sum()

    inputs = (weights, values, parameters, pending)
    with torch_compiler_warning_context():
        options = {"compile_threads": 1} if backend == "inductor" else None
        compiled = torch.compile(
            evaluate, fullgraph=True, backend=backend, options=options
        )
        actual = compiled(*inputs)
    expected = evaluate(*inputs)
    _close(actual, expected)
    leaves = (weights, values, *parameters.values(), pending)
    _close(
        torch.autograd.grad(actual[1], leaves), torch.autograd.grad(expected[1], leaves)
    )


@pytest.mark.parametrize("lanes, broadcast", ((0, True), (2, False), (2, True)))
@pytest.mark.parametrize("mapped", ("parameter", "buffer", "all"))
def test_vmap_independent_registered_parameters_and_buffers(lanes, broadcast, mapped):
    transform = _BroadcastAffineGate() if broadcast else _AffineGate()
    _, native = _connection((0.0,) * 4, transform=transform)
    projection, tensors = dn.func.make_functional_continuous_con(native)
    values = torch.tensor([[-0.8], [0.6]], dtype=DTYPE)
    dimensions = {
        name: 0
        if mapped == "all" or name == ("offset" if mapped == "buffer" else "scale")
        else None
        for name in tensors.transform_parameters
    }
    parameters = {
        name: value.unsqueeze(0)
        .expand(lanes, *value.shape)
        .clone()
        .detach()
        .requires_grad_()
        if dimensions[name] == 0
        else value
        for name, value in tensors.transform_parameters.items()
    }

    def evaluate(parameters):
        return projection.advance(
            tensors.weights, tensors.state, values, transform_parameters=parameters
        )

    actual = torch.vmap(evaluate, in_dims=(dimensions,))(parameters)
    if lanes:
        expected = [
            evaluate(
                {
                    name: value[lane] if dimensions[name] == 0 else value
                    for name, value in parameters.items()
                }
            )
            for lane in range(lanes)
        ]
        _close(actual, pytree.tree_map(lambda *leaves: torch.stack(leaves), *expected))
    else:
        assert all(value.shape[0] == 0 for value in pytree.tree_leaves(actual))
    mapped_parameters = [
        value for name, value in parameters.items() if dimensions[name] == 0
    ]
    gradients = torch.autograd.grad(actual[1].sum(), mapped_parameters)
    assert all(
        gradient.shape == value.shape
        for gradient, value in zip(gradients, mapped_parameters)
    )


def test_empty_scalar_parameter_vmap_matches_the_raw_torch_capability():
    _, native = _connection((0.0,) * 4, transform=_AffineGate())
    projection, tensors = dn.func.make_functional_continuous_con(native)
    values = torch.tensor([[-0.8], [0.6]], dtype=DTYPE)
    empty = torch.empty(0, dtype=DTYPE)
    parameters = {**tensors.transform_parameters, "scale": empty}
    dimensions = {name: 0 if name == "scale" else None for name in parameters}

    def evaluate(packet):
        return projection.advance(
            tensors.weights, tensors.state, values, transform_parameters=packet
        )

    try:
        torch.vmap(lambda scale: torch.ones(4, dtype=DTYPE) * scale)(empty)
    except IndexError as baseline_error:
        with pytest.raises(IndexError) as actual_error:
            torch.vmap(evaluate, in_dims=(dimensions,))(parameters)
        assert str(actual_error.value) == str(baseline_error)
        pytest.xfail(
            "PyTorch's empty mapped 0-D scalar multiplication raises IndexError"
        )
    else:
        actual = torch.vmap(evaluate, in_dims=(dimensions,))(parameters)
        torch.testing.assert_close(actual[1], values.new_empty(0, *projection.shape))


def test_forward_reverse_jacobians_include_registered_buffer_inputs():
    _, native = _connection((0.0,) * 4, transform=_AffineGate())
    projection, tensors = dn.func.make_functional_continuous_con(native)
    values = torch.tensor([[-0.8], [0.6]], dtype=DTYPE)

    def evaluate(weights, source, parameters):
        return projection.advance(
            weights, tensors.state, source, transform_parameters=parameters
        )[1]

    inputs = (tensors.weights, values, tensors.transform_parameters)
    _close(
        torch.func.jacfwd(evaluate, argnums=(0, 1, 2))(*inputs),
        torch.func.jacrev(evaluate, argnums=(0, 1, 2))(*inputs),
    )


class _InputMutation(torch.nn.Module):
    def forward(self, value):
        return value.add_(1)


class _BufferMutation(_AffineGate):
    def forward(self, value):
        self.offset.add_(1)
        return value + self.offset


class _BufferRebinding(_AffineGate):
    def forward(self, value):
        self.offset = self.offset + 1
        return value + self.offset


class _ParameterMutation(_AffineGate):
    def forward(self, value):
        self.scale.data.add_(1)
        return value * self.scale


class _ParameterRebinding(_AffineGate):
    def forward(self, value):
        self.scale = torch.nn.Parameter(self.scale + 1)
        return value * self.scale


class _PythonState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


class _UnregisteredTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.tensor(0.2, dtype=DTYPE)

    def forward(self, value):
        return value + self.hidden


@pytest.mark.parametrize(
    "factory",
    (
        _InputMutation,
        _BufferMutation,
        _BufferRebinding,
        _ParameterMutation,
        _ParameterRebinding,
        _PythonState,
        _UnregisteredTensor,
    ),
)
def test_impure_module_rejection_preserves_native_parameters_and_python_state(factory):
    _, native = _connection(transform=factory())
    original = _clone(native.transform.state_dict())
    original_carry = _clone(native.state_dict_for_checkpoint())
    calls = getattr(native.transform, "calls", None)
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional_continuous_con(native)
    _close(native.transform.state_dict(), original)
    _close(native.state_dict_for_checkpoint(), original_carry)
    assert getattr(native.transform, "calls", None) == calls


_MUTABLE_GLOBAL = [0.3]


def _mutable_global_transform(value):
    return value + _MUTABLE_GLOBAL[0]


@pytest.mark.parametrize("capture", ("tensor", "mutable", "global"))
def test_unregistered_tensor_and_mutable_function_dependencies_are_rejected(capture):
    dependency = torch.tensor(0.2, dtype=DTYPE) if capture == "tensor" else [0.2]
    if capture == "global":
        transform = _mutable_global_transform
    elif capture == "tensor":

        def transform(value):
            return value + dependency
    else:

        def transform(value):
            return value + dependency[0]

    _, native = _connection(transform=transform)
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional_continuous_con(native)
    assert _MUTABLE_GLOBAL == [0.3]
    if capture != "tensor":
        assert dependency == [0.2]


def _torch_random(value):
    return value + torch.rand_like(value)


def _python_random(value):
    return value + random.random()


def _numpy_random(value):
    return value + np.random.uniform()


_INITIAL_SEED_QUERY = torch.initial_seed
_RNG_STATE_QUERY = torch.get_rng_state


def _initial_seed_read(value):
    return value + (torch.initial_seed() % 97)


def _aliased_initial_seed_read(value):
    return value + (_INITIAL_SEED_QUERY() % 97)


def _rng_state_read(value):
    return value + torch.get_rng_state()[0]


def _aliased_rng_state_read(value):
    return value + _RNG_STATE_QUERY()[0]


def _nested_seed_read(value):
    def seed():
        return _INITIAL_SEED_QUERY()

    return value + (seed() % 97)


def _default_generator_read(value):
    return value + (torch.default_generator.initial_seed() % 97)


def _new_generator_read(value):
    return value + (torch.Generator().initial_seed() % 97)


@pytest.mark.parametrize(
    "transform",
    (
        _initial_seed_read,
        _aliased_initial_seed_read,
        _rng_state_read,
        _aliased_rng_state_read,
        _nested_seed_read,
        _default_generator_read,
        _new_generator_read,
    ),
)
def test_implicit_rng_state_reads_reject_before_query_execution(transform):
    _, native = _connection(transform=transform)
    rng_state = torch.random.get_rng_state().clone()
    query_codes = {_INITIAL_SEED_QUERY.__code__, _RNG_STATE_QUERY.__code__}
    calls = []
    previous_profile = sys.getprofile()

    def observe_queries(frame, event, argument):
        if event == "call" and frame.f_code in query_codes:
            calls.append(frame.f_code.co_name)
        elif (
            event == "c_call" and getattr(argument, "__name__", None) == "initial_seed"
        ):
            calls.append("initial_seed")

    try:
        sys.setprofile(observe_queries)
        with pytest.raises(dn.func.FunctionalizationError):
            dn.func.make_functional_continuous_con(native)
    finally:
        sys.setprofile(previous_profile)
    assert calls == []
    torch.testing.assert_close(torch.random.get_rng_state(), rng_state)


@pytest.mark.parametrize("transform", (_torch_random, _python_random, _numpy_random))
def test_implicit_randomness_rejects_and_restores_global_rngs(transform):
    _, native = _connection(transform=transform)
    torch_state = torch.random.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional_continuous_con(native)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


def test_authored_dynamic_import_rejects_before_hidden_callback_execution(monkeypatch):
    sentinel = ModuleType("dendra_transform_import_sentinel")
    calls = []

    def touch():
        calls.append("called")

    sentinel.touch = touch
    monkeypatch.setitem(sys.modules, sentinel.__name__, sentinel)

    def transform(value):
        from dendra_transform_import_sentinel import touch

        touch()
        return value + 1

    _, native = _connection(transform=transform)
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional_continuous_con(native)
    assert calls == []


@pytest.mark.parametrize(
    "transform",
    (
        lambda value: value.reshape(2, 2),
        lambda value: value.float(),
        lambda value: {"value": value},
    ),
)
def test_malformed_transform_output_rejects_lowering(transform):
    _, native = _connection(transform=transform)
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional_continuous_con(native)


@pytest.mark.parametrize(
    "mutation",
    (
        "instance_forward",
        "class_forward",
        "static_attribute",
        "parameter_shape",
        "new_buffer",
        "mode",
        "forward_hook",
        "call_dispatch",
        "tied_alias",
    ),
)
@pytest.mark.parametrize("compiled", (False, True))
def test_transform_configuration_and_schema_changes_invalidate_warm_calls(
    monkeypatch, mutation, compiled
):
    transform = _TiedGate() if mutation == "tied_alias" else _AffineGate()
    _, native = _connection(transform=transform)
    projection, tensors = dn.func.make_functional_continuous_con(native)
    values = torch.tensor([[-0.8], [0.6]], dtype=DTYPE)

    def evaluate(values):
        return projection.advance(
            tensors.weights,
            tensors.state,
            values,
            transform_parameters=tensors.transform_parameters,
        )

    run = (
        torch.compile(evaluate, backend="eager", fullgraph=True)
        if compiled
        else evaluate
    )
    run(values)
    if mutation == "instance_forward":
        monkeypatch.setattr(native.transform, "forward", lambda value: value)
    elif mutation == "class_forward":
        monkeypatch.setattr(_AffineGate, "forward", lambda self, value: value)
    elif mutation == "static_attribute":
        native.transform.gain = 2.0
    elif mutation == "parameter_shape":
        native.transform.scale = torch.nn.Parameter(torch.ones(2, dtype=DTYPE))
    elif mutation == "new_buffer":
        native.transform.register_buffer("fresh", torch.zeros((), dtype=DTYPE))
    elif mutation == "mode":
        native.transform.eval()
    elif mutation == "forward_hook":
        native.transform.register_forward_hook(lambda module, args, output: output)
    elif mutation == "call_dispatch":
        native.transform._call_impl = lambda value: value
    else:
        native.transform.right = torch.nn.Parameter(
            native.transform.left.detach().clone()
        )
    before = _clone(tensors.state)
    with pytest.raises((dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)):
        run(values)
    _close(tensors.state, before)


@pytest.mark.parametrize("compiled", (False, True))
def test_replaced_global_function_dependency_invalidates_existing_calls(
    monkeypatch, compiled
):
    _, native = _connection((0.0,) * 4, transform=_nested_transform)
    projection, tensors = dn.func.make_functional_continuous_con(native)
    values = torch.tensor([[-0.8], [0.6]], dtype=DTYPE)

    def evaluate(values):
        return projection.advance(tensors.weights, tensors.state, values)

    run = (
        torch.compile(evaluate, backend="eager", fullgraph=True)
        if compiled
        else evaluate
    )
    run(values)
    monkeypatch.setattr(sys.modules[__name__], "_signed_square", lambda value: value)
    with pytest.raises((dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)):
        run(values)


@pytest.mark.parametrize("compiled", (False, True))
def test_replaced_torch_python_routine_code_invalidates_existing_calls(
    monkeypatch, compiled
):
    _, native = _connection((0.0,) * 4, transform=torch.nn.ReLU())
    projection, tensors = dn.func.make_functional_continuous_con(native)
    values = torch.tensor([[-0.8], [0.6]], dtype=DTYPE)

    def evaluate(values):
        return projection.advance(tensors.weights, tensors.state, values)

    def replacement(value, inplace=False):
        return value

    run = (
        torch.compile(evaluate, backend="eager", fullgraph=True)
        if compiled
        else evaluate
    )
    run(values)
    monkeypatch.setattr(torch.nn.functional.relu, "__code__", replacement.__code__)
    with pytest.raises((dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)):
        run(values)


def test_poisoned_registered_map_descriptor_rejects_without_getter_execution(
    monkeypatch,
):
    _, native = _connection(transform=_AffineGate())
    calls = []

    def poison(module):
        calls.append("called")
        raise AssertionError("must not resolve authored storage metadata descriptors")

    monkeypatch.setattr(_AffineGate, "_parameters", property(poison), raising=False)
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional_continuous_con(native)
    assert calls == []


@pytest.mark.parametrize("invalid", ("missing", "extra", "dtype", "shape"))
def test_explicit_transform_packet_schema_rejects_without_mutation(invalid):
    _, native = _connection(transform=_AffineGate())
    projection, tensors = dn.func.make_functional_continuous_con(native)
    parameters = dict(tensors.transform_parameters)
    if invalid == "missing":
        parameters.pop("offset")
    elif invalid == "extra":
        parameters["extra"] = parameters["scale"]
    elif invalid == "dtype":
        parameters["offset"] = parameters["offset"].float()
    else:
        parameters["scale"] = parameters["scale"].reshape(1)
    before = _clone(tensors.state)
    with pytest.raises((TypeError, ValueError, dn.func.FunctionalizationError)):
        projection.advance(
            tensors.weights,
            tensors.state,
            torch.zeros(projection.source_shape, dtype=DTYPE),
            transform_parameters=parameters,
        )
    _close(tensors.state, before)
