"""Network-local receive frames retain ordered ownership and gradients."""

from collections.abc import Mapping

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra.models.mechanisms import Mechanism, State, Synapse
from dendra.models.mod import pas
from tests.func.test_network import (
    DT,
    DTYPE,
    _clone,
    _close,
    _loss,
    _mixed_network,
)
from tests.func.test_network_rollout import _mixed_targets

pytestmark = pytest.mark.cpu


class _FrameMemory(State):
    State.STATE("x")
    State.CARRY("mark")
    State.DERIVATIVE("x' = 0.0 * x")

    def state_defaults(self, v, values):
        return {"x": torch.zeros_like(v)}

    def initial_values(self, v, values):
        return {"mark": torch.zeros_like(v)}


class _InplaceReceive(Synapse):
    Synapse.STATE_BUNDLE(_FrameMemory)
    Synapse.CARRY("count")

    def initial_values(self, v, values):
        return {"count": torch.zeros_like(v)}

    def net_receive_values(self, weights, dt, values):
        # The ordinary binding already isolates authored in-place state work.
        # The Network refinement must retain that protection per transaction.
        values["x"].add_(weights)
        values["count"].add_(weights.square())
        mark = values["state_carry"]["_FrameMemory"]["mark"]
        mark.add_(weights * dt)
        return {
            "states": {"x": values["x"]},
            "carry": {"count": values["count"]},
            "state_carry": {"_FrameMemory": {"mark": mark}},
        }


class _ViewReceive(_InplaceReceive):
    def net_receive_values(self, weights, dt, values):
        return {"states": {"x": weights}}


class _IonAliasReceive(Synapse):
    Synapse.CARRY("nai")

    def initial_values(self, v, values):
        return {"nai": torch.zeros_like(v)}

    def net_receive_values(self, weights, dt, values):
        values["nai"].add_(weights)
        return {"carry": {"nai": values["nai"]}}


class _IonReader(Mechanism):
    Mechanism.USEION("na", read=["nai"])


def _population():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, CHECK_PURITY=False):
        model = dn.Population(N=2, C=1, dtype=DTYPE, integrator=dn.bwd_euler_sc())
        model.insert(pas)
        model.insert(_InplaceReceive.rename("left"))
        model.insert(_ViewReceive.rename("view"))
        model.train()
        model.initialize(DT)
        plan, tensors = dn.func.make_functional(model, dt=DT)
    return plan, tensors, plan._prepare_values(tensors.parameters, tensors.constants)


def _clone_count(graph):
    return sum(
        node.op == "call_function" and node.target is torch.ops.aten.clone.default
        for node in graph.graph.nodes
    )


@pytest.mark.parametrize("graph", (False, True), ids=("eager", "captured"))
def test_mixed_network_frames_preserve_order_all_gradients_and_input_state(graph):
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    state = _clone(tensors.state)
    relay = state["populations"]["cell"]["mechanism_buffers"]["relay"]
    level = relay["level"].detach().clone().requires_grad_()
    relay["level"] = level
    prepared = functional._prepare_values(tensors.parameters, tensors.constants)
    assert functional._binding_frame_admitted(state)
    before = _clone(state, detached=True)
    versions = [
        value._version for value in pytree.tree_leaves(state) if torch.is_tensor(value)
    ]

    def execute(parameters, values, incoming, *, frame):
        return functional._step_values(
            parameters, values, incoming, _binding_frame=frame
        )

    if graph:
        ordinary = make_fx(
            lambda p, v, s: execute(p, v, s, frame=False), pre_dispatch=True
        )(tensors.parameters, prepared, state)
        framed = make_fx(
            lambda p, v, s: execute(p, v, s, frame=True), pre_dispatch=True
        )(tensors.parameters, prepared, state)
        assert _clone_count(framed) < _clone_count(ordinary)
    else:

        def ordinary(p, v, s):
            return execute(p, v, s, frame=False)

        def framed(p, v, s):
            return execute(p, v, s, frame=True)

    expected_state = actual_state = state
    # Include both delayed projection histories, so every connection and
    # transform source contributes a real gradient rather than an unused slot.
    for index in range(7):
        expected = ordinary(tensors.parameters, prepared, expected_state)
        actual = framed(tensors.parameters, prepared, actual_state)
        _close(actual, expected)
        if index == 0:
            observed = actual[0]["populations"]["cell"]["mechanism_buffers"]["relay"]
            torch.testing.assert_close(observed["level"], torch.full_like(level, 1.5))
        expected_state, actual_state = expected[0], actual[0]
    targets = _mixed_targets(tensors.parameters, level)
    actual_gradients = torch.autograd.grad(_loss(actual[0]), targets, retain_graph=True)
    expected_gradients = torch.autograd.grad(_loss(expected[0]), targets)
    _close(actual_gradients, expected_gradients)
    assert all(torch.isfinite(value).all() for value in actual_gradients)
    _close(state, before)
    assert [
        value._version for value in pytree.tree_leaves(state) if torch.is_tensor(value)
    ] == versions


def test_target_copies_preserve_prior_saved_values_and_nested_carry_gradients():
    plan, tensors, prepared = _population()
    state = _clone(tensors.state)
    shape = plan._receiver_shapes["left"]
    x = torch.full(shape, 0.3, dtype=DTYPE, requires_grad=True)
    count = torch.full(shape, 0.2, dtype=DTYPE, requires_grad=True)
    mark = torch.full(shape, -0.1, dtype=DTYPE, requires_grad=True)
    state["mechanisms"]["left"]["x"] = x
    state["mechanism_buffers"]["left"]["count"] = count
    state["state_buffers"]["left"]["_FrameMemory"]["mark"] = mark
    first_weight = torch.full(shape, 0.4, dtype=DTYPE, requires_grad=True)
    second_weight = torch.full(shape, 0.7, dtype=DTYPE, requires_grad=True)
    first = plan._receive_values(
        tensors.parameters, prepared, state, "left", first_weight
    )

    def values(frame):
        return (
            frame["mechanisms"]["left"]["x"],
            frame["mechanism_buffers"]["left"]["count"],
            frame["state_buffers"]["left"]["_FrameMemory"]["mark"],
        )

    first_values = values(first)
    frozen = _clone(first, detached=True)
    first_loss = sum(value.square().sum() for value in first_values)
    first_versions = [value._version for value in first_values]
    second = plan._receive_values(
        tensors.parameters,
        prepared,
        first,
        "left",
        second_weight,
        _binding_indices=plan._receiver_state_indices["left"],
    )
    _close(first, frozen)
    assert [value._version for value in first_values] == first_versions
    assert second["integrator"]["v"] is first["integrator"]["v"]
    assert second["mechanisms"]["view"]["x"] is first["mechanisms"]["view"]["x"]
    for current, previous in zip(values(second), first_values, strict=True):
        assert not torch._C._is_alias_of(current, previous)
    for leaf in plan._state_layout:
        incoming, outgoing = state, first
        for name in leaf.public_path:
            incoming, outgoing = incoming[name], outgoing[name]
        assert not torch._C._is_alias_of(incoming, outgoing)

    actual_loss = first_loss + sum(value.square().sum() for value in values(second))
    a = (x + first_weight, count + first_weight.square(), mark + DT * first_weight)
    b = (
        a[0] + second_weight,
        a[1] + second_weight.square(),
        a[2] + DT * second_weight,
    )
    reference = sum(value.square().sum() for value in (*a, *b))
    inputs = (x, count, mark, first_weight, second_weight)
    _close(
        torch.autograd.grad(actual_loss, inputs), torch.autograd.grad(reference, inputs)
    )


def test_target_copy_closure_protects_carry_aliased_to_an_ion_checkpoint_leaf():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, CHECK_PURITY=False):
        model = dn.Population(N=2, C=1, dtype=DTYPE, integrator=dn.bwd_euler_sc())
        model.insert(pas)
        model.insert(_IonReader)
        model.insert(_IonAliasReceive.rename("na_ion"))
        model.train()
        model.initialize(DT)
        plan, tensors = dn.func.make_functional(model, dt=DT)
    prepared = plan._prepare_values(tensors.parameters, tensors.constants)
    receiver = plan._receivers["na_ion"]
    ion_index = next(
        index
        for index, entry in enumerate(plan._state_layout)
        if entry.public_path == ("ions", "na", "nai")
    )
    assert ion_index not in receiver._state_indices
    assert ion_index in receiver._copy_indices
    shape = plan._receiver_shapes["na_ion"]
    state = _clone(tensors.state)
    initial = torch.full(shape, 0.3, dtype=DTYPE, requires_grad=True)
    state["ions"]["na"]["nai"] = initial
    first_weight = torch.full(shape, 0.4, dtype=DTYPE, requires_grad=True)
    second_weight = torch.full(shape, 0.7, dtype=DTYPE, requires_grad=True)
    first = plan._receive_values(
        tensors.parameters, prepared, state, "na_ion", first_weight
    )
    first_value = first["ions"]["na"]["nai"]
    first_loss = first_value.square().sum()
    version = first_value._version
    expected = plan._receive_values(
        tensors.parameters, prepared, first, "na_ion", second_weight
    )
    actual = plan._receive_values(
        tensors.parameters,
        prepared,
        first,
        "na_ion",
        second_weight,
        _binding_indices=plan._receiver_state_indices["na_ion"],
    )
    _close(actual, expected)
    _close(first_value, initial + first_weight)
    assert first_value._version == version
    value = actual["ions"]["na"]["nai"]
    assert not torch._C._is_alias_of(value, first_value)
    a = initial + first_weight
    reference = a.square().sum() + (a + second_weight).square().sum()
    inputs = initial, first_weight, second_weight
    _close(
        torch.autograd.grad(first_loss + value.square().sum(), inputs),
        torch.autograd.grad(reference, inputs),
    )


@pytest.mark.parametrize("view", ("slice", "expanded", "negative"))
def test_previous_target_views_are_normalized_before_a_different_receiver(view):
    plan, tensors, prepared = _population()
    state = _clone(tensors.state)
    raw = torch.arange(7, dtype=DTYPE, requires_grad=True)
    if view == "slice":
        payload = raw[1:5:2].reshape(2, 1)
    elif view == "expanded":
        payload = raw[1:2].expand(2).reshape(2, 1)
    else:
        payload = torch._neg_view(raw[1:3].reshape(2, 1))
    first = plan._receive_values(tensors.parameters, prepared, state, "view", payload)
    assert torch._C._is_alias_of(first["mechanisms"]["view"]["x"], raw)
    weights = torch.zeros(plan._receiver_shapes["left"], dtype=DTYPE)
    expected = plan._receive_values(
        tensors.parameters, prepared, first, "left", weights
    )
    actual = plan._receive_values(
        tensors.parameters,
        prepared,
        first,
        "left",
        weights,
        _binding_indices=(
            plan._receiver_state_indices["view"] | plan._receiver_state_indices["left"]
        ),
    )
    _close(actual, expected)
    a = actual["mechanisms"]["view"]["x"]
    b = expected["mechanisms"]["view"]["x"]
    assert (a.stride(), a.storage_offset(), a.is_conj(), a.is_neg()) == (
        b.stride(),
        b.storage_offset(),
        b.is_conj(),
        b.is_neg(),
    )
    assert not torch._C._is_alias_of(a, raw)
    _close(
        torch.autograd.grad(a.square().sum(), raw, retain_graph=True),
        torch.autograd.grad(b.square().sum(), raw),
    )


@pytest.mark.parametrize(
    ("execution", "inference"),
    (("scan", True), ("while", True), ("scan", False)),
    ids=("scan", "while", "scan-training"),
)
def test_stock_parameter_state_uses_one_network_engine_across_continuation(
    execution, inference
):
    from dendra.func._state_projection import _plain_state_tree

    source, native = _mixed_network(), _mixed_network()
    program = dn.compile(
        source,
        dt=DT,
        steps=4,
        execution=execution,
        inference=inference,
        initialization="snapshot",
        backend="aot_eager",
        trainable=(),
    )
    initial = _clone(program.initial_state, detached=True)
    voltage = torch.nn.Parameter(initial["populations"]["cell"]["integrator"]["v"])
    initial["populations"]["cell"]["integrator"]["v"] = voltage
    if not inference:
        native_voltage = native.cell.v.detach().clone().requires_grad_()
        native.cell.v = native_voltage
    assert not _plain_state_tree(initial)
    assert _plain_state_tree(initial, allow_parameters=True)
    assert program.functional._binding_frame_admitted(initial)
    with torch.set_grad_enabled(not inference):
        first, auxiliary = program(initial)
        for _ in range(4):
            native.step()
        _close(first, program.functional.extract(native).state)
        assert type(first["populations"]["cell"]["integrator"]["v"]) is torch.Tensor
        kernel = program.kernel._plain_kernel
        assert kernel is not None and len(kernel._kernels) == 1
        assert program.kernel._legacy_plain_kernel is None
        assert program.functional._binding_frame_admitted(first)
        final, _ = program(auxiliary["continuation"])
        for _ in range(4):
            native.step()
        _close(final, program.functional.extract(native).state)
        if not inference:
            gradient = torch.autograd.grad(
                final["populations"]["cell"]["integrator"]["v"].square().sum(), voltage
            )[0]
            expected = torch.autograd.grad(
                native.cell.v.square().sum(), native_voltage
            )[0]
            _close(gradient, expected)
            assert torch.count_nonzero(gradient) > 0
    assert program.kernel._plain_kernel is kernel
    assert len(kernel._kernels) == 1
    assert program.kernel._legacy_plain_kernel is None


@pytest.mark.parametrize("override", ("class", "instance"))
def test_parameter_clone_overrides_keep_the_network_binding_fallback(
    override, monkeypatch
):
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    state = _clone(tensors.state)
    voltage = torch.nn.Parameter(state["populations"]["cell"]["integrator"]["v"])
    state["populations"]["cell"]["integrator"]["v"] = voltage
    assert functional._binding_frame_admitted(state)

    def observed_clone(*args, **kwargs):
        raise AssertionError("admission must not call authored clone methods")

    monkeypatch.setattr(
        torch.nn.Parameter if override == "class" else voltage, "clone", observed_clone
    )
    assert not functional._binding_frame_admitted(state)


@pytest.mark.parametrize(
    "kind", ("mapping", "subclass", "function_mode", "dispatch_mode")
)
def test_network_frame_admission_preserves_observable_input_fallbacks(kind):
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    state = tensors.state
    assert functional._binding_frame_admitted(state)
    if kind == "mapping":
        reads = []

        class Observed(Mapping):
            def __getitem__(self, key):
                reads.append(key)
                return state[key]

            def __iter__(self):
                raise AssertionError("admission must not enumerate custom mappings")

            def __len__(self):
                raise AssertionError("admission must not inspect custom mappings")

        assert not functional._binding_frame_admitted(Observed())
        assert reads == []
    elif kind == "subclass":

        class Value(torch.Tensor):
            pass

        changed = _clone(state)
        voltage = changed["populations"]["cell"]["integrator"]["v"]
        changed["populations"]["cell"]["integrator"]["v"] = torch.Tensor._make_subclass(
            Value, voltage, False
        )
        assert not functional._binding_frame_admitted(changed)
    elif kind == "function_mode":

        class Mode(TorchFunctionMode):
            def __torch_function__(self, function, types, args=(), kwargs=None):
                return function(*args, **(kwargs or {}))

        with Mode():
            assert not functional._binding_frame_admitted(state)
    else:

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                return function(*args, **(kwargs or {}))

        with Mode():
            assert not functional._binding_frame_admitted(state)


@pytest.mark.parametrize(
    "override", ("mapping", "source", "ion", "material", "current")
)
def test_changed_binding_and_handler_hooks_retain_the_ordinary_path(
    monkeypatch, override
):
    model = _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    assert functional._binding_frame_admitted(tensors.state)
    if override == "mapping":
        owner, name = functional.populations["cell"], "_mapping"
    elif override == "source":
        owner, name = functional, "_source_value"
    else:
        owner = model.cell.mech
        name = {
            "ion": "read_from_ions",
            "material": "read_from_materials",
            "current": "_bind_ion_current_fields",
        }[override]
    monkeypatch.setattr(owner, name, getattr(owner, name))
    assert not functional._binding_frame_admitted(tensors.state)
