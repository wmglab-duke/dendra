"""Capture-time Parameter normalization preserves ordinary loop execution."""

from __future__ import annotations

import io
import sys
from copy import deepcopy
from types import MethodType

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._hop_inputs import (
    _isolate_dense_lifted_inputs,
    _LiftedViewIsolation,
    _parameter_input_isolation,
)
from dendra.func._scan import _CompiledScanPopulationChunk
from dendra.func._tensor_contracts import capture_schema
from dendra.func._tensor_tree_while import _CompiledTensorTreeWhile
from dendra.models.mod import pas

pytestmark = pytest.mark.cpu
DT = 0.025
STEPS = 4


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


def _activity():
    from torch._dynamo import utils

    return deepcopy(utils.counters), len(utils.get_compilation_metrics())


def _snapshot(tree):
    values, spec = pytree.tree_flatten(tree)
    metadata, copies = [], []
    for value in values:
        if torch.is_tensor(value):
            metadata.append(
                (
                    id(value),
                    value._version,
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.storage_offset(),
                    value.requires_grad,
                    value.is_leaf,
                    type(value),
                )
            )
            copies.append(value.detach().clone())
        else:
            metadata.append(value)
            copies.append(value)
    return spec, metadata, copies


def _unchanged(tree, snapshot):
    observed = _snapshot(tree)
    assert observed[:2] == snapshot[:2]
    for actual, expected in zip(observed[2], snapshot[2], strict=True):
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected


def _owned(result, inputs):
    outputs = [value for value in pytree.tree_leaves(result) if torch.is_tensor(value)]
    sources = [value for value in pytree.tree_leaves(inputs) if torch.is_tensor(value)]
    for index, value in enumerate(outputs):
        assert all(
            not torch._C._is_alias_of(value, other)
            for other in (*outputs[index + 1 :], *sources)
        )


def _bindings(value):
    return (
        {"p": value, "alias": value},
        {"g": value.square() + 0.2, "same": value, "unused": value.new_tensor(3.0)},
    )


def _scalar_oracle(used, parameters, prepared, state, carries, tape, *, leaf=True):
    # Independent recurrence: no authored function or loop adapter is called.
    coefficient = 3 * parameters["p"] if used else prepared["g"]
    coefficient = (2.0 if leaf else 11.0) * coefficient
    value, energy, count = state["x"], carries["energy"], carries["count"]
    frames = []
    for sample in tape["u"]:
        value = value + coefficient * sample
        energy = energy + value.square()
        count = count + 1
        frames.append(value)
    return {"x": value}, {"energy": energy, "count": count}, {"x": torch.stack(frames)}


@pytest.mark.parametrize("used", (False, True), ids=("unused", "used-aliases"))
def test_while_restored_parameter_reuses_backend_and_keeps_raw_capture_policy(used):
    def body(parameters, prepared, state, carries, sample):
        # Observe the raw source before captured operands are normalized.
        policy = 2.0 if parameters["p"].is_leaf else 11.0
        coefficient = (
            parameters["p"] + parameters["alias"] + prepared["same"]
            if used
            else prepared["g"]
        )
        value = state["x"] + policy * coefficient * sample["u"]
        return (
            {"x": value},
            {
                "energy": carries["energy"] + value.square(),
                "count": carries["count"] + 1,
            },
        ), {"x": value}

    engine = _CompiledTensorTreeWhile(
        body,
        steps=STEPS,
        compile_options={"backend": "aot_eager", "fullgraph": True},
        device="cpu",
    )
    parameter = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    restored = parameter.detach().clone().requires_grad_(parameter.requires_grad)
    initial = {"x": torch.tensor(0.5, dtype=torch.float64)}
    carries = {
        "energy": torch.tensor(0.13, dtype=torch.float64),
        "count": torch.tensor(0, dtype=torch.int32),
    }
    tape = {"u": torch.tensor([0.1, -0.2, 0.4, 0.3], dtype=torch.float64)}
    global_rng = torch.get_rng_state().clone()
    with torch.no_grad():
        first_bindings, second_bindings = _bindings(parameter), _bindings(restored)
        assert capture_schema(first_bindings) == capture_schema(second_bindings)
        arguments = (*first_bindings, initial, carries, tape)
        frozen = _snapshot(arguments)
        first = engine(*arguments)
        expected = _scalar_oracle(used, *arguments)
        torch.testing.assert_close(first, expected, rtol=2e-11, atol=2e-12)
        _unchanged(arguments, frozen)
        _owned(first, arguments)
        admitted = _activity()
        assert len(engine._kernels) == 1

        # A restored leaf supplies an independent suffix owner while tied
        # parameter/prepared occurrences retain their explicit aliases.
        assert (
            first_bindings[0]["p"]
            is first_bindings[0]["alias"]
            is first_bindings[1]["same"]
        )
        assert (
            second_bindings[0]["p"]
            is second_bindings[0]["alias"]
            is second_bindings[1]["same"]
        )
        arguments = (*second_bindings, first[0], first[1], tape)
        frozen = _snapshot(arguments)
        actual = engine(*arguments)
        expected = _scalar_oracle(used, *arguments)
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
        _unchanged(arguments, frozen)
        _owned(actual, (arguments, first))
        assert _activity() == admitted
        assert len(engine._kernels) == 1
        repeated = engine(*arguments)
        torch.testing.assert_close(repeated, expected, rtol=2e-11, atol=2e-12)
        assert _activity() == admitted

        restored.add_(0.17)
        arguments = (*_bindings(restored), actual[0], actual[1], tape)
        frozen = _snapshot(arguments)
        updated = engine(*arguments)
        torch.testing.assert_close(
            updated, _scalar_oracle(used, *arguments), rtol=2e-11, atol=2e-12
        )
        _unchanged(arguments, frozen)
        assert _activity() == admitted
        assert len(engine._kernels) == 1

    # A different authored leafness policy must still receive a new capture.
    with torch.enable_grad():
        nonleaf = restored * 1.0
    assert not nonleaf.is_leaf
    with torch.no_grad():
        arguments = (*_bindings(nonleaf), initial, carries, tape)
        actual = engine(*arguments)
        torch.testing.assert_close(
            actual, _scalar_oracle(used, *arguments, leaf=False), rtol=2e-11, atol=2e-12
        )
        assert len(engine._kernels) == 2
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


def _population():
    model = dn.SingleCompartment(
        N=1,
        C=2,
        dtype=torch.float64,
        v_init=-62.0,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(pas, g=1e-4, e=-65.0)
    model.initialize()
    model.train()
    return model


@pytest.mark.parametrize("slot", ("raw_source", "prepared_dt"))
def test_population_inference_scan_reuses_restored_captured_operands(slot):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    key = next(name for name in tensors.parameters if name.endswith("g_param"))
    value = tensors.parameters[key] if slot == "raw_source" else tensors.constants["dt"]
    parameter = torch.nn.Parameter(
        value.detach().clone(), requires_grad=value.requires_grad
    )
    restored = parameter.detach().clone().requires_grad_(parameter.requires_grad)
    drive = torch.linspace(0.0, 0.2e-9, STEPS * 2, dtype=torch.float64).reshape(
        STEPS, 1, 2
    )
    engine = _CompiledScanPopulationChunk(
        functional,
        steps=STEPS,
        compile_options={"backend": "aot_eager", "fullgraph": True},
    )
    global_rng = torch.get_rng_state().clone()
    original = _snapshot((tensors.parameters, tensors.constants, tensors.state))

    def arguments(source, state):
        parameters = dict(tensors.parameters)
        if slot == "raw_source":
            parameters[key] = source
        prepared = functional.prepare(parameters, tensors.constants).values
        if slot == "prepared_dt":
            prepared = {
                **prepared,
                "integrator": {**prepared["integrator"], "dt": source},
            }
        return parameters, prepared, state, None, drive

    with torch.no_grad():
        first_arguments = arguments(parameter, tensors.state)
        second_arguments = arguments(restored, tensors.state)
        assert capture_schema(first_arguments) == capture_schema(second_arguments)
        admitted = None
        for source in (parameter, restored, restored):
            current = arguments(source, tensors.state)
            frozen = _snapshot(current)
            actual = engine(*current)
            parameters, prepared, state, ve, intra = current
            expected = functional._transition_rollout_values(
                parameters, prepared, state, ve, intra, STEPS
            )
            torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
            _unchanged(current, frozen)
            assert len(engine._kernels) == 1
            if admitted is None:
                admitted = _activity()
            else:
                assert _activity() == admitted
        restored.mul_(1.2)
        current = arguments(restored, actual[0])
        frozen = _snapshot(current)
        continued = engine(*current)
        torch.testing.assert_close(
            continued,
            functional._transition_rollout_values(*current, STEPS),
            rtol=2e-10,
            atol=2e-11,
        )
        _unchanged(current, frozen)
        assert _activity() == admitted
        assert len(engine._kernels) == 1
    _unchanged((tensors.parameters, tensors.constants, tensors.state), original)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_population_portable_inference_continuation_reuses_parameter_bindings(
    execution,
):
    program = dn.compile(
        _population(),
        dt=DT,
        steps=STEPS,
        execution=execution,
        inference=True,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
    )
    parameters = dict(program.parameters)
    key = next(name for name in parameters if name.endswith("g_param"))
    parameters[key] = torch.nn.Parameter(
        parameters[key].detach().clone(), requires_grad=parameters[key].requires_grad
    )
    original = _snapshot((parameters, program.constants, program.initial_state))
    prefix, first = program(parameters=parameters)
    admitted = _activity()
    stream = io.BytesIO()
    torch.save(program.export_continuation(first["continuation"]), stream)
    stream.seek(0)
    restored = program.restore_continuation(torch.load(stream, weights_only=True))
    assert type(restored.parameters[key]) is torch.Tensor
    assert restored.parameters[key] is not parameters[key]
    suffix, last = program(restored)
    assert _activity() == admitted
    repeated, replay = program(restored)
    assert _activity() == admitted
    with torch.no_grad():
        expected, reference = program.functional.prepare_and_rollout(
            restored.parameters,
            restored.constants,
            restored.state,
            steps=STEPS,
            callbacks=program.callbacks,
            callback_state=restored.callback_state,
        )
    torch.testing.assert_close(suffix, expected, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(repeated, expected, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(
        last["callbacks"].outputs,
        reference["callbacks"].outputs,
        rtol=2e-10,
        atol=2e-11,
    )
    torch.testing.assert_close(
        replay["callbacks"].outputs, last["callbacks"].outputs, rtol=0, atol=0
    )
    time = program.initial_state["clock"]["t"]
    for _ in range(STEPS):
        time = time + DT
    torch.testing.assert_close(prefix["clock"]["t"], time, rtol=0, atol=0)
    for _ in range(STEPS):
        time = time + DT
    torch.testing.assert_close(suffix["clock"]["t"], time, rtol=0, atol=0)
    assert first["callbacks"]["trace"]["v"].shape[0] == STEPS + 1
    assert last["callbacks"]["trace"]["v"].shape[0] == STEPS
    _unchanged((parameters, program.constants, program.initial_state), original)


def _isolation(kernel):
    # Capture materializers retain the selected callable, never example values.
    bindings = dict(zip(kernel.__code__.co_freevars, kernel.__closure__, strict=True))
    return bindings["isolate_views"].cell_contents


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_ordinary_tensor_capture_keeps_original_isolation_and_warm_path(
    execution, monkeypatch
):
    if execution == "scan":
        functional, tensors = dn.func.make_functional(_population(), dt=DT)
        parameters = pytree.tree_map_only(
            torch.Tensor,
            lambda value: value.detach().clone().requires_grad_(value.requires_grad),
            tensors.parameters,
        )
        prepared = functional.prepare(parameters, tensors.constants).values
        engine = _CompiledScanPopulationChunk(
            functional,
            steps=STEPS,
            compile_options={"backend": "aot_eager", "fullgraph": True},
        )
        arguments = parameters, prepared, tensors.state, None, None

        def reference():
            return functional._transition_rollout_values(*arguments, STEPS)
    else:

        def body(parameters, prepared, state, carries, sample):
            value = state["x"] + parameters["p"] * sample["u"]
            return ({"x": value}, carries), {"x": value}

        engine = _CompiledTensorTreeWhile(
            body,
            steps=STEPS,
            compile_options={"backend": "aot_eager", "fullgraph": True},
            device="cpu",
        )
        arguments = (
            {"p": torch.tensor(0.7, dtype=torch.float64)},
            {},
            {"x": torch.tensor(0.5, dtype=torch.float64)},
            {},
            {"u": torch.tensor([0.1, -0.2, 0.4, 0.3], dtype=torch.float64)},
        )

        def reference():
            value, frames = arguments[2]["x"], []
            for drive in arguments[4]["u"]:
                value = value + arguments[0]["p"] * drive
                frames.append(value)
            return {"x": value}, {}, {"x": torch.stack(frames)}

    assert all(
        type(value) is torch.Tensor
        for value in pytree.tree_leaves(arguments)
        if torch.is_tensor(value)
    )
    frozen = _snapshot(arguments)
    with torch.no_grad():
        expected = reference()
        actual = engine(*arguments)
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
        assert len(engine._kernels) == 1
        selected = _isolation(next(iter(engine._kernels.values())))
        # Ordinary captures use precisely the pre-existing isolation callable.
        # They must not acquire a Parameter wrapper or runtime proof branch.
        assert type(selected) is _LiftedViewIsolation
        assert selected.captured_inputs is False

        def forbidden_factory(*_args, **_kwargs):
            pytest.fail("normal runtime invoked Parameter capture factory")

        for name in (
            "dendra.func._hop_inputs",
            "dendra.func._scan",
            "dendra.func._tensor_tree_while",
        ):
            monkeypatch.setattr(
                sys.modules[name],
                "_parameter_input_isolation",
                forbidden_factory,
                raising=False,
            )
        admitted = _activity()
        for _ in range(2):
            repeated = engine(*arguments)
            torch.testing.assert_close(repeated, expected, rtol=2e-10, atol=2e-11)
            assert _activity() == admitted
            assert _isolation(next(iter(engine._kernels.values()))) is selected
    _unchanged(arguments, frozen)


def test_parameter_normalization_keeps_unused_values_ties_and_live_gradients():
    parameter = torch.nn.Parameter(torch.tensor([0.2, 0.7], dtype=torch.float64))
    plain = torch.tensor([-0.3, 0.5], dtype=torch.float64)
    arguments = {
        "used": parameter,
        "unused": parameter,
        "plain": plain,
        "optional": None,
        "tag": "retained",
    }
    base = _LiftedViewIsolation.from_usage(
        {"used": True, "unused": False, "plain": False, "optional": False, "tag": False}
    )
    isolate = _parameter_input_isolation(base, arguments)
    assert isolate is not base
    frozen = _snapshot(arguments)
    with torch.enable_grad():
        normalized = isolate(arguments)
        assert type(normalized["used"]) is torch.Tensor
        assert normalized["used"] is normalized["unused"]
        assert normalized["plain"] is plain
        assert normalized["optional"] is None and normalized["tag"] == "retained"
        assert torch._C._is_alias_of(normalized["used"], parameter)
        (normalized["used"] + normalized["unused"]).sum().backward()
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 2.0))
    _unchanged(arguments, frozen)

    # The view observes updates without a copy, and a later restored Tensor is
    # already canonical. Unused sources remain real operands in both calls.
    with torch.no_grad():
        parameter.add_(0.17)
        torch.testing.assert_close(normalized["unused"], parameter, rtol=0, atol=0)
        restored = parameter.detach().clone().requires_grad_(parameter.requires_grad)
        restored_arguments = {**arguments, "used": restored, "unused": restored}
        result = isolate(restored_arguments)
    assert result is restored_arguments
    assert result["used"] is result["unused"] is restored
    assert result["plain"] is plain


def test_parameter_normalization_factory_retains_exact_tensor_base():
    value = torch.tensor([0.2, 0.7], dtype=torch.float64)
    arguments = {"used": value, "unused": value, "tag": "kept"}
    base = _LiftedViewIsolation.from_usage(
        {"used": True, "unused": False, "tag": False}
    )
    assert _parameter_input_isolation(base, arguments) is base
    assert base(arguments) is arguments


@pytest.mark.parametrize("used", (False, True), ids=("unused", "used"))
@pytest.mark.parametrize(
    "owner", ("tensor_instance", "tensor_class", "parameter_class")
)
def test_parameter_frame_clone_overrides_keep_original_path(monkeypatch, used, owner):
    tensor = torch.tensor([0.2, 0.7], dtype=torch.float64)
    parameter = torch.nn.Parameter(torch.tensor([0.4, 0.9], dtype=torch.float64))
    observed = parameter if owner == "parameter_class" else tensor
    arguments = {"observed": observed, "parameter": parameter}
    base = _LiftedViewIsolation.from_usage({"observed": used, "parameter": used})
    isolate = _parameter_input_isolation(base, arguments)
    original = torch.Tensor.clone
    frozen = _snapshot(arguments)
    failure = RuntimeError("authored parameter-frame clone")
    events, fail = [], False

    def clone(source, **kwargs):
        if source is observed:
            events.append((id(source), kwargs))
            if fail:
                raise failure
        return original(source, **kwargs) + 3

    with monkeypatch.context() as scoped:
        if owner == "tensor_instance":
            scoped.setattr(observed, "clone", MethodType(clone, observed))
        else:
            scoped.setattr(
                torch.nn.Parameter if owner == "parameter_class" else torch.Tensor,
                "clone",
                clone,
            )
        # Use the original dense boundary as the oracle, including any
        # DeviceContext redispatch of a class-level clone override.
        expected = _isolate_dense_lifted_inputs(base(arguments))
        expected_events = list(events)
        assert expected_events
        events.clear()
        normalized = isolate(arguments)
        assert normalized is arguments
        assert events == []
        actual = _isolate_dense_lifted_inputs(normalized)
        assert events == expected_events
        events.clear()
        fail = True
        assert isolate(arguments) is arguments
        assert events == []
        with pytest.raises(RuntimeError) as caught:
            _isolate_dense_lifted_inputs(arguments)
        assert caught.value is failure
        assert events == [(id(observed), {"memory_format": torch.preserve_format})]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _unchanged(arguments, frozen)


@pytest.mark.parametrize("used", (False, True), ids=("unused", "used"))
@pytest.mark.parametrize("kind", ("function_mode", "dispatch_mode"))
def test_parameter_frame_modes_keep_original_operands_and_errors(used, kind):
    observed = torch.tensor([0.2, 0.7], dtype=torch.float64)
    parameter = torch.nn.Parameter(torch.tensor([0.4, 0.9], dtype=torch.float64))
    arguments = {"observed": observed, "parameter": parameter}
    base = _LiftedViewIsolation.from_usage({"observed": used, "parameter": used})
    isolate = _parameter_input_isolation(base, arguments)
    frozen = _snapshot(arguments)
    failure = RuntimeError("observed parameter-frame clone")
    events, fail = [], False
    clone = (
        torch.Tensor.clone if kind == "function_mode" else torch.ops.aten.clone.default
    )
    alias = torch.ops.aten.alias.default

    def observe(function, args):
        if function is alias and args[0] is parameter:
            events.append("parameter alias")
        if function is clone and args[0] is observed:
            events.append("observed clone")
            if fail:
                raise failure

    if kind == "function_mode":

        class Mode(TorchFunctionMode):
            def __torch_function__(self, function, types, args=(), kwargs=None):
                observe(function, args)
                return function(*args, **(kwargs or {}))
    else:

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                observe(function, args)
                return function(*args, **(kwargs or {}))

    with Mode():
        normalized = isolate(arguments)
        assert normalized is arguments and events == []
        actual = _isolate_dense_lifted_inputs(normalized)
        assert events == ["observed clone"]
        events.clear()
        fail = True
        assert isolate(arguments) is arguments and events == []
        with pytest.raises(RuntimeError) as caught:
            _isolate_dense_lifted_inputs(arguments)
        assert caught.value is failure and events == ["observed clone"]
    torch.testing.assert_close(actual, arguments, rtol=0, atol=0)
    _owned(actual, arguments)
    _unchanged(arguments, frozen)


@pytest.mark.parametrize("used", (False, True), ids=("unused", "used"))
def test_parameter_view_normalization_preserves_offset_stride_and_ownership(used):
    parameter = torch.nn.Parameter(torch.arange(9, dtype=torch.float64)[1::2])
    arguments = {"parameter": parameter, "tag": "kept"}
    base = _LiftedViewIsolation.from_usage({"parameter": used, "tag": False})
    isolate = _parameter_input_isolation(base, arguments)
    frozen = _snapshot(arguments)
    with torch.enable_grad():
        normalized = isolate(arguments)
        value = normalized["parameter"]
        assert type(value) is torch.Tensor
        assert value.shape == parameter.shape
        assert value.stride() == parameter.stride()
        assert value.storage_offset() == parameter.storage_offset()
        assert torch._C._is_alias_of(value, parameter) is not used
        assert normalized["tag"] == "kept"
        torch.testing.assert_close(value, parameter, rtol=0, atol=0)
        value.square().sum().backward()
    torch.testing.assert_close(parameter.grad, 2 * parameter, rtol=0, atol=0)
    _unchanged(arguments, frozen)
