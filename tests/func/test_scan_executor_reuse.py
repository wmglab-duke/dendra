"""Equivalent scan executors share compilation without hiding authored views."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import FunctionalizationError
from dendra.func._carry_packing import _PackedCarryLayout
from dendra.func._hop_inputs import _LiftedViewIsolation
from dendra.func._runners import _checkpoint_tensor_tree
from dendra.func._scan_tree import (
    _capture_tensor_step,
    _CompiledTensorTreeScan,
    _tensor_samples,
)
from dendra.func._tensor_contracts import capture_schema
from dendra.helpers import MEMBRANE_ASSEMBLY

pytestmark = pytest.mark.cpu


def test_capture_types_do_not_call_authored_metaclass_hash_or_equality():
    observations = []

    class Meta(type(torch.Tensor)):
        def __hash__(cls):
            observations.append("hash")
            return type.__hash__(cls)

        def __eq__(cls, other):
            observations.append("equality")
            return cls is other

    class Subclass(torch.Tensor, metaclass=Meta):
        pass

    value = torch.ones((), dtype=torch.float64).as_subclass(Subclass)
    observations.clear()
    first = capture_schema({"value": value})
    second = capture_schema({"value": value})
    # Existing tensor-property dispatch can inspect its class. Constructed
    # capture keys must not add authored metaclass operations during lookup.
    observations.clear()
    assert first == second
    assert hash(first) == hash(second)
    assert observations == []


def test_parameter_class_recapture_reuses_a_type_independent_executor():
    def body(parameters, prepared, state, carries, sample):
        value = state["v"] + parameters["p"].square() * sample["drive"]
        return ({"v": value}, carries), {"v": value}

    engine = _engine(body)
    parameter = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
    restored = parameter.detach().clone().requires_grad_()
    for source in (parameter, restored):
        initial = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
        tape = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
        final, _, emitted = engine(
            {"p": source}, {}, {"v": initial}, {}, {"drive": tape}
        )
        expected = initial + source.square() * tape.cumsum(0)
        torch.testing.assert_close(final["v"], expected[-1])
        torch.testing.assert_close(emitted["v"], expected)
        actual_gradients = torch.autograd.grad(
            final["v"].square() + emitted["v"].square().sum(),
            (source, initial, tape),
        )
        reference_gradients = torch.autograd.grad(
            expected[-1].square() + expected.square().sum(), (source, initial, tape)
        )
        torch.testing.assert_close(actual_gradients, reference_gradients)
    # Authored class observations require separate raw captures. Equivalent
    # numerical FX programs still share the expensive compiled boundary.
    assert len(engine._kernels) == 2
    assert len(_executables(engine)) == 1


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            yield
    finally:
        torch.set_num_threads(previous)


def _activity():
    from torch._dynamo import utils

    return deepcopy(utils.counters)


def _body(parameters, prepared, state, carries, sample):
    drive = sample["drive"]
    value = torch.tanh(
        parameters["a"] * state["v"] + (parameters["b"] * drive.real).sum()
    )
    return (
        {"v": value},
        {"energy": carries["energy"] + value.square()},
    ), {"v": value}


def _engine(body=_body):
    return _CompiledTensorTreeScan(
        body,
        steps=2,
        compile_options={"backend": "aot_eager", "fullgraph": True},
        device="cpu",
    )


def _source(length=6):
    def leaf(value):
        return torch.tensor(value, dtype=torch.float64, requires_grad=True)

    return (
        {"a": leaf(0.73), "b": leaf(0.29)},
        {"v": leaf(-0.21)},
        {"energy": leaf(0.17)},
        torch.linspace(-0.4, 0.7, length, dtype=torch.float64, requires_grad=True),
    )


def _reference(parameters, state, carries, tape):
    # Independent recurrence; this does not call the captured authored body.
    value, energy = state["v"], carries["energy"]
    frames = []
    for frame in tape:
        value = (parameters["a"] * value + parameters["b"] * frame.real.sum()).tanh()
        energy = energy + value.square()
        frames.append(value)
    return {"v": value}, {"energy": energy}, {"v": torch.stack(frames)}


def _join(parts):
    return (
        *parts[-1][:2],
        {"v": torch.cat([part[2]["v"] for part in parts])},
    )


def _loss(result):
    state, carries, emitted = result
    weights = torch.arange(1, emitted["v"].numel() + 1, dtype=torch.float64)
    return (
        state["v"].square()
        + 0.3 * carries["energy"]
        + (weights * emitted["v"].square()).sum()
    )


def _targets(source):
    parameters, state, carries, tape = source
    return (*parameters.values(), state["v"], carries["energy"], tape)


def _assert_gradients(actual, expected, *, allow_unused=False):
    actual_loss, actual_targets = actual
    expected_loss, expected_targets = expected
    gradients = torch.autograd.grad(
        actual_loss, actual_targets, retain_graph=True, allow_unused=allow_unused
    )
    reference = torch.autograd.grad(
        expected_loss, expected_targets, retain_graph=True, allow_unused=allow_unused
    )
    for observed, wanted in zip(gradients, reference, strict=True):
        if wanted is None:
            assert observed is None
        else:
            assert observed is not None and torch.isfinite(observed).all()
            torch.testing.assert_close(observed, wanted, rtol=2e-10, atol=2e-11)
    return gradients


def _executables(engine):
    return {id(entry[0]) for entry in engine._kernels.values()}


def test_offset_drive_chunks_share_executor_and_full_training_history():
    source, reference = _source(), _source()
    parameters, state, carries, tape = source
    frozen = tape.detach().clone()
    version = tape._version
    engine = _engine()
    parts = []
    admitted = None
    for start in (0, 2, 4):
        selected = tape[start : start + 2]
        assert selected.storage_offset() == start
        result = engine(parameters, {}, state, carries, {"drive": selected})
        parts.append(result)
        state, carries = result[:2]
        # Prime backward on the first call as well: a forward-only counter
        # snapshot would miss a new AOT backward specialization.
        torch.autograd.grad(_loss(result), _targets(source), retain_graph=True)
        if admitted is None:
            admitted = _activity()
        else:
            assert _activity() == admitted
    actual = _join(parts)
    expected = _reference(*reference)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    gradients = _assert_gradients(
        (_loss(actual), _targets(source)), (_loss(expected), _targets(reference))
    )
    assert all(value.abs().sum() > 0 for value in gradients)
    assert _activity() == admitted
    assert len(engine._kernels) == 3  # Each raw view is still authored separately.
    assert len(engine._executors) == len(_executables(engine)) == 1
    assert tape._version == version
    torch.testing.assert_close(tape, frozen, rtol=0, atol=0)


def test_scan_executor_reuse_respects_omitted_default_backend():
    source, reference = _source(), _source()
    parameters, state, carries, tape = source
    options = {"fullgraph": True}
    engine = _CompiledTensorTreeScan(
        _body, steps=2, compile_options=options, device="cpu"
    )
    parts = []
    for start in (0, 2, 4):
        result = engine(
            parameters, {}, state, carries, {"drive": tape[start : start + 2]}
        )
        parts.append(result)
        state, carries = result[:2]
    actual = _join(parts)
    expected = _reference(*reference)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    _assert_gradients(
        (_loss(actual), _targets(source)), (_loss(expected), _targets(reference))
    )
    assert options == {"fullgraph": True}
    assert len(engine._kernels) == 3
    assert len(engine._executors) == len(_executables(engine)) == 1


@pytest.mark.parametrize("observer", ("offset", "absolute_view"))
def test_metadata_sensitive_sample_programs_keep_separate_executors(observer):
    def body(parameters, prepared, state, carries, sample):
        drive = sample["drive"]
        if observer == "offset":
            increment = drive + drive.storage_offset()
        else:
            increment = drive.as_strided((), (), 0)
        value = state["v"] + increment
        return ({"v": value}, carries), {"v": value}

    tape = torch.tensor([0.2, 0.7, 1.1, 1.6], dtype=torch.float64)
    state = {"v": tape.new_tensor(0.0)}
    engine = _engine(body)
    control = None
    if observer == "absolute_view":
        from torch._dynamo.backends.debugging import aot_eager

        def backend(graph, examples):
            return aot_eager(graph, examples)

        control = _engine(body)
        control._options["backend"] = backend
    original_metadata = tape.stride(), tape.storage_offset(), tape._version
    with torch.no_grad():
        for start in (0, 2):
            selected = tape[start : start + 2]
            actual = engine({}, {}, state, {}, {"drive": selected})
            if observer == "offset":
                # A captured sample-metadata branch belongs to this raw frame.
                expected = selected.sum() + 2 * selected[0].storage_offset()
            else:
                # Stock AOT scan owns/materializes its samples. Preserve that
                # physical-view contract rather than assuming native-loop
                # access to the entire original tape's storage prefix.
                expected = control({}, {}, state, {}, {"drive": selected})[0]["v"]
            torch.testing.assert_close(actual[0]["v"], expected)
    assert len(engine._kernels) == len(_executables(engine)) == 2
    assert (tape.stride(), tape.storage_offset(), tape._version) == original_metadata
    if control is not None:
        assert not control._executors
        assert len(control._kernels) == len(_executables(control)) == 2


def _view(source, kind):
    if kind == "strided":
        return source.reshape(6, 4)[:, ::2]
    if kind == "expanded":
        return source[2:8].reshape(6, 1).expand(6, 2)
    if kind == "overlap":
        return source.as_strided((6, 2), (1, 1), 2)
    raise AssertionError("unknown sample layout")


@pytest.mark.parametrize("kind", ("strided", "expanded", "overlap"))
def test_noncanonical_sample_layouts_keep_values_views_and_base_gradients(kind):
    source, reference = _source(24), _source(24)
    parameters, state, carries, base = source
    tape, ref_tape = _view(base, kind), _view(reference[3], kind)
    metadata = (tape.stride(), tape.storage_offset(), tape.is_conj(), tape.is_neg())
    frozen, version = base.detach().clone(), base._version
    engine, parts = _engine(), []
    for start in (0, 2, 4):
        result = engine(
            parameters, {}, state, carries, {"drive": tape[start : start + 2]}
        )
        parts.append(result)
        state, carries = result[:2]
    actual = _join(parts)
    expected = _reference(*reference[:3], ref_tape)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    _assert_gradients(
        (_loss(actual), _targets(source)), (_loss(expected), _targets(reference))
    )
    assert len(engine._kernels) == len(_executables(engine)) == 3
    assert metadata == (
        tape.stride(),
        tape.storage_offset(),
        tape.is_conj(),
        tape.is_neg(),
    )
    assert base._version == version
    torch.testing.assert_close(base, frozen, rtol=0, atol=0)


@pytest.mark.parametrize("flag", ("conjugate", "negative"))
def test_flagged_samples_keep_existing_isolation_and_decline_pooling(flag):
    from dendra.func._scan_executor import _executor_key

    base = torch.linspace(-0.4, 0.7, 24, dtype=torch.float64, requires_grad=True)
    reference = base.detach().clone().requires_grad_()
    if flag == "conjugate":
        tape = torch.complex(base, 0.3 * base).reshape(6, 4).conj()[2:4]
        expected = torch.complex(reference, 0.3 * reference).reshape(6, 4).conj()[2:4]
    else:
        tape = torch._neg_view(base.reshape(6, 4))[2:4]
        expected = torch._neg_view(reference.reshape(6, 4))[2:4]
    isolate = _LiftedViewIsolation.from_usage({"drive": True}, captured_inputs=True)
    selected = isolate({"drive": tape})["drive"]
    assert selected.is_conj() == tape.is_conj()
    assert selected.is_neg() == tape.is_neg()
    assert selected.storage_offset() == tape.storage_offset()
    assert selected.stride() == tape.stride()
    assert not torch._C._is_alias_of(selected, tape)
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)
    _assert_gradients(
        (selected.real.square().sum(), (base,)),
        (expected.real.square().sum(), (reference,)),
    )
    # The stock scan HOP has no Conjugate/Negative dispatch. Exercise admission
    # and the existing differentiable view boundary without adding capabilities.
    graph, signature = _key_example()
    layout = signature[0]
    initial = layout.pack(({"v": base.new_tensor(0.1)}, {}))
    samples, sample_layout = _tensor_samples({"drive": tape}, 2, "cpu")
    flagged_signature = (
        layout,
        capture_schema(({}, {}, initial, {}, samples)),
        sample_layout,
        MEMBRANE_ASSEMBLY.value,
        False,
    )
    assert (
        _executor_key(
            graph, flagged_signature, {"backend": "aot_eager", "fullgraph": True}
        )
        is None
    )


def test_sample_gradient_flags_and_none_positions_remain_distinct():
    def body(parameters, prepared, state, carries, sample):
        drive = sample["drive"]
        if drive is None:
            drive = sample["alternate"]
        factor = 2.0 if drive.requires_grad else 5.0
        value = state["v"] + factor * drive
        return ({"v": value}, carries), {"v": value}

    engine = _engine(body)
    for train in (False, True):
        for alternate in (False, True):
            tape = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=train)
            sample = {
                "drive": None if alternate else tape,
                "alternate": tape if alternate else None,
            }
            initial = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
            actual = engine({}, {}, {"v": initial}, {}, sample)
            expected = initial + (2.0 if train else 5.0) * tape.sum()
            torch.testing.assert_close(actual[0]["v"], expected)
            targets = (initial, tape) if train else (initial,)
            _assert_gradients(
                (actual[0]["v"].square(), targets), (expected.square(), targets)
            )
    assert len(engine._kernels) == len(_executables(engine)) == 4


def test_identical_graph_code_keeps_distinct_tensor_constant_owners(monkeypatch):
    import dendra.func._scan_tree as scan_module

    choices = [torch.tensor(0.25, dtype=torch.float64)]
    graphs = []
    capture = scan_module._capture_tensor_step

    def observe(*args, **kwargs):
        graph = capture(*args, **kwargs)
        graphs.append(graph)
        return graph

    monkeypatch.setattr(scan_module, "_capture_tensor_step", observe)

    def body(parameters, prepared, state, carries, sample):
        value = state["v"] + sample["drive"] + choices[0]
        return ({"v": value}, carries), {"v": value}

    engine = _engine(body)
    tape = torch.tensor([0.2, 0.7, 1.1, 1.6, 0.4, -0.2], dtype=torch.float64)
    owners = []
    with torch.no_grad():
        for start, bias in ((0, 0.25), (2, 0.25), (4, -0.7)):
            choices[0] = tape.new_tensor(bias)
            owners.append(choices[0])
            actual = engine(
                {},
                {},
                {"v": tape.new_tensor(0.0)},
                {},
                {"drive": tape[start : start + 2]},
            )
            torch.testing.assert_close(
                actual[0]["v"], tape[start : start + 2].sum() + 2 * bias
            )
    assert len(graphs) == 3
    assert all(graph.code == graphs[0].code for graph in graphs)
    assert owners[0] is not owners[1]
    torch.testing.assert_close(owners[0], owners[1], rtol=0, atol=0)
    assert len(engine._kernels) == len(_executables(engine)) == 3


def _key_example():
    # Match the engine's public operand roles without compiling a hostile mode.
    state = {"v": torch.tensor(0.1, dtype=torch.float64)}
    layout = _PackedCarryLayout.from_tree((state, {}))
    initial = layout.pack((state, {}))
    samples, sample_layout = _tensor_samples(
        {"drive": torch.tensor([0.2, 0.7], dtype=torch.float64)}, 2, "cpu"
    )
    signature = (
        layout,
        capture_schema(({}, {}, initial, {}, samples)),
        sample_layout,
        MEMBRANE_ASSEMBLY.value,
        False,
    )
    constant = torch.tensor(0.3, dtype=torch.float64)

    def authored(parameters, prepared, state, carries, invariant, sample):
        value = state["v"] + sample["drive"] + constant
        return ({"v": value}, carries), {"v": value}

    graph = _capture_tensor_step(
        authored,
        ({}, {}, state, {}, {}, {"drive": samples[0][0]}),
        grad_enabled=False,
    )
    return graph, signature


@pytest.mark.parametrize(
    "change", ("hook", "instance_forward", "class_forward", "stale_graph")
)
def test_executor_admission_rejects_changed_graph_execution(monkeypatch, change):
    from dendra.func._scan_executor import _executor_key

    graph, signature = _key_example()
    options = {"backend": "aot_eager", "fullgraph": True}
    assert _executor_key(graph, signature, options) is not None

    def authored(*args, **kwargs):
        raise AssertionError("admission must not execute authored code")

    if change == "hook":
        handle = graph.register_forward_hook(authored)
        try:
            assert _executor_key(graph, signature, options) is None
        finally:
            handle.remove()
        assert _executor_key(graph, signature, options) is not None
        return
    if change == "instance_forward":
        monkeypatch.setattr(graph, "forward", authored)
    elif change == "class_forward":
        monkeypatch.setattr(type(graph), "forward", authored)
    else:
        node = next(
            node
            for node in graph.graph.nodes
            if node.op == "call_function" and node.target is torch.ops.aten.add.Tensor
        )
        node.target = torch.ops.aten.sub.Tensor
    assert _executor_key(graph, signature, options) is None


def test_executor_constants_keep_mutation_versions_in_the_key():
    from dendra.func._scan_executor import _executor_key

    graph, signature = _key_example()
    options = {"backend": "aot_eager", "fullgraph": True}
    original = _executor_key(graph, signature, options)
    assert original is not None
    constant = next(iter(graph.buffers()))
    with torch.no_grad():
        constant.add_(0.25)
    changed = _executor_key(graph, signature, options)
    assert changed is not None and changed != original


@pytest.mark.parametrize("marker", ("guarded", "unguarded"))
def test_compiler_buffer_markers_keep_constant_identity_without_method_overrides(
    marker,
):
    from dendra.func._scan_executor import _executor_key

    graph, signature = _key_example()
    options = {"backend": "aot_eager", "fullgraph": True}
    original = _executor_key(graph, signature, options)
    assert original is not None
    constant = next(iter(graph.buffers()))
    constant._dynamo_static_input_type = marker
    assert _executor_key(graph, signature, options) == original

    def authored():
        raise AssertionError("admission must not invoke an authored tensor method")

    constant.stride = authored
    assert _executor_key(graph, signature, options) is None


def test_executor_admission_retains_custom_output_pytree_path():
    from collections import namedtuple

    from dendra.func._scan_executor import _executor_key

    output = namedtuple("AuthoredOutput", "state emission")
    _unused, signature = _key_example()

    def body(parameters, prepared, state, carries, invariant, sample):
        value = state["v"] + sample["drive"]
        return output(value, value.square())

    graph = _capture_tensor_step(
        body,
        ({}, {}, {"v": torch.tensor(0.1)}, {}, {}, {"drive": torch.tensor(0.2)}),
        grad_enabled=False,
    )
    assert (
        _executor_key(graph, signature, {"backend": "aot_eager", "fullgraph": True})
        is None
    )


@pytest.mark.parametrize("custom", ("backend", "option"))
def test_executor_admission_keeps_unsupported_compile_options(custom):
    from dendra.func._scan_executor import _executor_key

    class Observable:
        def __hash__(self):
            raise AssertionError("admission must not hash authored configuration")

    graph, signature = _key_example()
    options = {"backend": "aot_eager", "fullgraph": True}
    assert _executor_key(graph, signature, options) is not None
    if custom == "backend":
        options["backend"] = Observable()
    else:
        options["options"] = {"authored": Observable()}
    assert _executor_key(graph, signature, options) is None


@pytest.mark.parametrize("authored", ("dispatch_mode", "subclass"))
def test_executor_admission_declines_authored_tensor_dispatch(authored):
    from dendra.func._scan_executor import _executor_key

    graph, signature = _key_example()
    options = {"backend": "aot_eager", "fullgraph": True}
    assert _executor_key(graph, signature, options) is not None
    events = []
    if authored == "subclass":

        class Value(torch.Tensor):
            @classmethod
            def __torch_function__(cls, function, types, args=(), kwargs=None):
                events.append(function)
                raise AssertionError("authored Tensor operation")

        name = next(iter(graph._buffers))
        graph._buffers[name] = torch.Tensor._make_subclass(
            Value, graph._buffers[name], False
        )
        assert _executor_key(graph, signature, options) is None
    else:

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                events.append(function)
                raise AssertionError("authored dispatch mode")

        with Mode():
            assert _executor_key(graph, signature, options) is None
    assert events == []


@pytest.mark.parametrize("boundary", ("live", "detach"))
def test_reused_executor_preserves_checkpoint_suffix_hooks_and_output_guard(boundary):
    source, reference = _source(4), _source(4)
    parameters, state, carries, tape = source
    engine = _engine()
    marker, events = object(), []

    def pack(value):
        events.append("pack")
        return marker, value.detach()

    def unpack(saved):
        token, value = saved
        assert token is marker
        events.append("unpack")
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        prefix = engine(parameters, {}, state, carries, {"drive": tape[:2]})
        ref_prefix = _reference(*reference[:3], reference[3][:2])
        state, carries = prefix[:2]
        ref_state, ref_carries = ref_prefix[:2]
        if boundary == "detach":
            state, carries = pytree.tree_map_only(
                torch.Tensor, lambda value: value.detach(), (state, carries)
            )
            ref_state, ref_carries = pytree.tree_map_only(
                torch.Tensor, lambda value: value.detach(), (ref_state, ref_carries)
            )

        def execute(parameters, state, carries, drive):
            return engine(parameters, {}, state, carries, {"drive": drive})

        actual = _checkpoint_tensor_tree(
            execute, (parameters, state, carries, tape[2:])
        )
        expected = _reference(reference[0], ref_state, ref_carries, reference[3][2:])
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
        gradients = _assert_gradients(
            (_loss(actual), _targets(source)),
            (_loss(expected), _targets(reference)),
            allow_unused=True,
        )
        if boundary == "detach":
            assert gradients[2] is gradients[3] is None
            torch.testing.assert_close(gradients[-1][:2], torch.zeros_like(tape[:2]))
        assert "pack" in events and "unpack" in events
    assert len(engine._executors) == (2 if boundary == "detach" else 1)
    # A shared callable must still apply the public first-order-only boundary.
    guarded = engine(parameters, {}, source[1], source[2], {"drive": tape[2:]})
    with pytest.raises(FunctionalizationError, match="first-order reverse-mode"):
        torch.autograd.grad(_loss(guarded), parameters["a"], create_graph=True)


def test_connected_network_explicit_inputs_reach_shared_scan_executor(monkeypatch):
    import dendra as dn
    from tests.func.test_network import (
        _RAW_WEIGHT,
        DT,
        DTYPE,
        _clone,
        _close,
    )
    from tests.func.test_network import _loss as network_loss
    from tests.func.test_network import (
        _mixed_network,
    )

    model = _mixed_network()
    event = next(iter(model.synapses))
    continuous = next(iter(model.continuous_synapses))
    names = (
        f"synapses.{event}.{_RAW_WEIGHT}",
        f"continuous_synapses.{continuous}.{_RAW_WEIGHT}",
    )
    runner = dn.compile(
        model,
        dt=DT,
        steps=6,
        chunk_steps=2,
        execution="scan",
        inference=False,
        trainable=names,
        backend="aot_eager",
    )
    state = _clone(runner.initial_state, detached=True)
    voltage = state["populations"]["cell"]["integrator"]["v"]
    initial_voltage = voltage.clone().requires_grad_()
    state["populations"]["cell"]["integrator"]["v"] = initial_voltage
    base = (
        torch.linspace(0.1, 0.3, 6 * voltage.numel(), dtype=DTYPE) * dn.units.nA
    ).requires_grad_()
    tape = base.reshape(6, *voltage.shape)
    inputs = {"cell": dn.func.Inputs(intra=tape)}
    frozen = _clone((state, inputs), detached=True)
    original_version = base._version
    expected, _ = runner.functional.prepare_and_rollout(
        runner.parameters, runner.constants, state, inputs=inputs, steps=6
    )

    calls = []
    execute = _CompiledTensorTreeScan.__call__

    def observe(engine, *args, **kwargs):
        result = execute(engine, *args, **kwargs)
        calls.append((engine, _activity()))
        return result

    monkeypatch.setattr(_CompiledTensorTreeScan, "__call__", observe)
    actual, _ = runner(state, inputs=inputs)
    _close(actual, expected)
    # Callback-free Network chunks choose between the admitted receiver frame
    # and the legacy binding transaction, each with its own scan adapter.
    engines = [
        engine
        for engine in (runner.kernel._plain_kernel, runner.kernel._legacy_plain_kernel)
        if engine is not None
    ]
    assert len(engines) == 1
    engine = engines[0]
    assert len(calls) == 3 and all(owner is engine for owner, _ in calls)
    assert all(activity == calls[0][1] for _, activity in calls)
    assert len(engine._kernels) == 3
    assert len(engine._executors) == len(_executables(engine)) == 1
    targets = (base, initial_voltage, *(runner.get_parameter(name) for name in names))
    gradients = _assert_gradients(
        (network_loss(actual), targets), (network_loss(expected), targets)
    )
    assert all(gradient.abs().sum() > 0 for gradient in gradients)
    assert actual["synapses"] and actual["continuous_synapses"]
    _close((state, inputs), frozen)
    assert base._version == original_version
