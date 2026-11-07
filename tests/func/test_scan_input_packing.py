"""Owned scan inputs retain training history across public continuation layouts."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._hop_inputs import _LiftedViewIsolation
from dendra.func._scan_tree import _CompiledTensorTreeScan

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _warnings():
    with torch_compiler_warning_context():
        yield


def _leaves(tree):
    return [value for value in pytree.tree_leaves(tree) if torch.is_tensor(value)]


def _snapshot(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _history_clone(tree):
    return pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), tree)


def _leaf_copy(tree):
    return pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.detach().clone().requires_grad_(value.requires_grad),
        tree,
    )


def _owned(result, originals):
    values = _leaves(result)
    for index, value in enumerate(values):
        assert value.is_contiguous()
        assert all(
            not torch._C._is_alias_of(value, other)
            for other in (*values[index + 1 :], *_leaves(originals))
        )


def _body(parameters, prepared, state, carries, sample):
    value = (
        parameters["a"] * state["x"]
        + 0.1 * state["same"]
        + parameters["b"] * sample["u"]
    )
    updated = {"x": value, "same": value.view_as(value), "none": None}
    carry = {"energy": carries["energy"] + value.square(), "none": None}
    return (updated, carry), {"x": value, "same": value.view_as(value)}


def _source(*, train=True):
    def scalar(value):
        return torch.tensor(value, dtype=torch.float64, requires_grad=train)

    x = scalar(0.7)
    return {
        "parameters": {"a": scalar(0.8), "b": scalar(0.4)},
        "state": {"x": x, "same": x, "none": None},
        "carries": {"energy": scalar(0.13), "none": None},
        "tapes": [
            {
                "u": torch.tensor(values, dtype=torch.float64, requires_grad=train),
                "unused": None,
            }
            for values in ([0.2, -0.1], [0.4, 0.3], [-0.2, 0.5])
        ],
    }


def _oracle(parameters, state, carries, tapes):
    # Independent Python recurrence, without the authored body or scan helpers.
    x, same, energy = state["x"], state["same"], carries["energy"]
    frames = []
    for tape in tapes:
        for index in range(2):
            x = parameters["a"] * x + 0.1 * same + parameters["b"] * tape["u"][index]
            same = x
            energy = energy + x.square()
            frames.append(x)
    values = torch.stack(frames)
    return (
        {"x": x, "same": same, "none": None},
        {"energy": energy, "none": None},
        {"x": values, "same": values},
    )


def _join(parts):
    return (
        parts[-1][0],
        parts[-1][1],
        {name: torch.cat([part[2][name] for part in parts]) for name in ("x", "same")},
    )


def _loss(result):
    state, carries, emitted = result
    weights = torch.arange(1, emitted["x"].numel() + 1, dtype=torch.float64)
    return (
        state["x"].square()
        + carries["energy"]
        + (weights * emitted["x"].square()).sum()
        + 0.3 * emitted["same"].square().sum()
    )


def _targets(source):
    return (
        *source["parameters"].values(),
        source["state"]["x"],
        source["carries"]["energy"],
        *(tape["u"] for tape in source["tapes"]),
    )


def _engine(monkeypatch, body=_body):
    from torch._dynamo import utils

    captures = []
    engine = _CompiledTensorTreeScan(
        body,
        steps=2,
        compile_options={"backend": "aot_eager", "fullgraph": True},
        device="cpu",
    )
    before = utils.counters["stats"]["unique_graphs"]

    def compilation_count():
        return utils.counters["stats"]["unique_graphs"] - before

    original = engine._capture

    def capture(*args, **kwargs):
        captures.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_capture", capture)
    return engine, compilation_count, captures


@pytest.mark.parametrize("boundary", ("raw", "clone"))
def test_training_continuations_reuse_compilation_and_preserve_full_bptt(
    monkeypatch, boundary
):
    source, reference = _source(), _source()
    before = _snapshot(source)
    leaves = _leaves(source)
    versions = [value._version for value in leaves]
    flags = [(value.requires_grad, value.is_leaf) for value in leaves]
    engine, compilation_count, captures = _engine(monkeypatch)
    state, carries = source["state"], source["carries"]
    parts = []
    backend_count = None
    for index, tape in enumerate(source["tapes"]):
        if index == 1 and boundary == "clone":
            state, carries = _history_clone((state, carries))
        actual = engine(source["parameters"], {}, state, carries, tape)
        if backend_count is None:
            backend_count = compilation_count()
            assert backend_count > 0
        assert compilation_count() == backend_count
        assert len(captures) == len(engine._kernels) == 1
        _owned(actual, (source, parts))
        parts.append(actual)
        state, carries = actual[:2]
        assert state["x"].requires_grad and not state["x"].is_leaf
    actual = _join(parts)
    expected = _oracle(
        reference["parameters"],
        reference["state"],
        reference["carries"],
        reference["tapes"],
    )
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    saved = _snapshot(parts)
    gradients = torch.autograd.grad(_loss(actual), _targets(source))
    oracle = torch.autograd.grad(_loss(expected), _targets(reference))
    for value, wanted in zip(gradients, oracle, strict=True):
        assert torch.isfinite(value).all() and torch.count_nonzero(wanted) > 0
        torch.testing.assert_close(value, wanted, rtol=2e-10, atol=2e-11)
    assert compilation_count() == backend_count
    assert len(captures) == len(engine._kernels) == 1
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    torch.testing.assert_close(parts, saved, rtol=0, atol=0)
    assert [value._version for value in leaves] == versions
    assert [(value.requires_grad, value.is_leaf) for value in leaves] == flags
    with torch.no_grad():
        parts[-1][0]["x"].add_(100)
    torch.testing.assert_close(parts[:-1], saved[:-1], rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)


@pytest.mark.parametrize("boundary", ("detach", "restored_leaves"))
def test_training_suffix_gradients_follow_independent_boundary_ownership(
    monkeypatch, boundary
):
    source, reference = _source(), _source()
    engine, _graphs, _captures = _engine(monkeypatch)
    first = engine(
        source["parameters"], {}, source["state"], source["carries"], source["tapes"][0]
    )
    reference_first = _oracle(
        reference["parameters"],
        reference["state"],
        reference["carries"],
        reference["tapes"][:1],
    )
    before = _snapshot((source, first))
    if boundary == "detach":
        state, carries = pytree.tree_map_only(
            torch.Tensor, lambda value: value.detach(), first[:2]
        )
        ref_state, ref_carries = pytree.tree_map_only(
            torch.Tensor, lambda value: value.detach(), reference_first[:2]
        )
        parameters, ref_parameters = source["parameters"], reference["parameters"]
        targets, wanted_targets = _targets(source), _targets(reference)
        disconnected = (2, 3, 4)  # Original initial state/carry and prefix tape.
    else:
        # Portable restoration supplies new owned leaves with their saved flags;
        # the public export/restore format is covered by continuation tests.
        parameters, state, carries = _leaf_copy((source["parameters"], *first[:2]))
        ref_parameters, ref_state, ref_carries = _leaf_copy(
            (reference["parameters"], *reference_first[:2])
        )
        restored = (*parameters.values(), state["x"], state["same"], carries["energy"])
        ref_restored = (
            *ref_parameters.values(),
            ref_state["x"],
            ref_state["same"],
            ref_carries["energy"],
        )
        assert all(value.requires_grad and value.is_leaf for value in restored)
        targets = (
            *restored,
            *(tape["u"] for tape in source["tapes"][1:]),
            *_targets(source)[:5],
        )
        wanted_targets = (
            *ref_restored,
            *(tape["u"] for tape in reference["tapes"][1:]),
            *_targets(reference)[:5],
        )
        disconnected = tuple(range(len(restored) + 2, len(targets)))
    parts = []
    for tape in source["tapes"][1:]:
        part = engine(parameters, {}, state, carries, tape)
        _owned(part, (source, first, parts))
        parts.append(part)
        state, carries = part[:2]
    actual = _join(parts)
    expected = _oracle(ref_parameters, ref_state, ref_carries, reference["tapes"][1:])
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    gradients = torch.autograd.grad(_loss(actual), targets, allow_unused=True)
    oracle = torch.autograd.grad(_loss(expected), wanted_targets, allow_unused=True)
    for index, (value, wanted) in enumerate(zip(gradients, oracle, strict=True)):
        if index in disconnected:
            assert value is wanted is None
        else:
            assert value is not None and wanted is not None
            assert torch.isfinite(value).all() and torch.count_nonzero(wanted) > 0
            torch.testing.assert_close(value, wanted, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close((source, first), before, rtol=0, atol=0)


def test_no_grad_scan_normalizes_offset_alias_inference_and_continued_inputs(
    monkeypatch,
):
    source = _source(train=False)
    engine, compilation_count, captures = _engine(monkeypatch)
    previous = []
    backend_count = None
    with torch.no_grad():
        for kind in ("ordinary", "offset", "independent", "inference", "continued"):
            if kind == "continued":
                state, carries = previous[0][:2]
            elif kind == "inference":
                with torch.inference_mode():
                    state, carries = _history_clone(
                        (source["state"], source["carries"])
                    )
            elif kind == "offset":
                storage = torch.full((7,), 0.7, dtype=torch.float64)
                state = {"x": storage[3], "same": storage[3], "none": None}
                carries = source["carries"]
            elif kind == "independent":
                state, carries = _history_clone((source["state"], source["carries"]))
            else:
                state, carries = source["state"], source["carries"]
            operands = source["parameters"], {}, state, carries, source["tapes"][0]
            saved = _snapshot(operands)
            versioned = [
                value for value in _leaves(operands) if not value.is_inference()
            ]
            versions = [value._version for value in versioned]
            actual = engine(*operands)
            if backend_count is None:
                backend_count = compilation_count()
                assert backend_count > 0
            assert compilation_count() == backend_count, kind
            assert len(captures) == len(engine._kernels) == 1, kind
            expected = _oracle(
                source["parameters"], state, carries, source["tapes"][:1]
            )
            torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
            _owned(actual, (operands, previous))
            torch.testing.assert_close(operands, saved, rtol=0, atol=0)
            assert [value._version for value in versioned] == versions
            previous.append(actual)
    assert compilation_count() == backend_count
    assert len(captures) == len(engine._kernels) == 1


def test_scan_retains_raw_invariant_view_metadata_specialization_and_gradients(
    monkeypatch,
):
    def body(parameters, prepared, state, carries, sample):
        value = parameters["probe"]
        updated = {"x": state["x"] + value.sum() + value.storage_offset()}
        return (updated, carries), updated

    engine, compilation_count, captures = _engine(monkeypatch, body)
    upstream = torch.ones(8, dtype=torch.float64, requires_grad=True)
    initial = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    tape = {"u": torch.zeros(2, dtype=torch.float64)}
    for offset in (0, 2):
        value = upstream[offset : offset + 4 : 2]
        actual = engine({"probe": value}, {}, {"x": initial}, {}, tape)
        reference = upstream.detach().clone().requires_grad_()
        ref_initial = initial.detach().clone().requires_grad_()
        increment = reference[offset : offset + 4 : 2].sum() + offset
        frames = ref_initial + torch.arange(1, 3, dtype=torch.float64) * increment
        torch.testing.assert_close(actual[0]["x"], frames[-1])
        torch.testing.assert_close(actual[2]["x"], frames)
        gradients = torch.autograd.grad(
            actual[0]["x"].square() + actual[2]["x"].square().sum(), (upstream, initial)
        )
        oracle = torch.autograd.grad(
            frames[-1].square() + frames.square().sum(), (reference, ref_initial)
        )
        torch.testing.assert_close(gradients, oracle, rtol=2e-10, atol=2e-11)
        assert value.stride() == (2,) and value.storage_offset() == offset
        assert value.requires_grad and not value.is_leaf
    assert len(captures) == compilation_count() == 2


def test_scan_host_pack_honors_custom_clone_values_and_autograd():
    calls = []

    def clone(state):
        calls.append(None)
        return {"x": 2 * state["x"].clone(memory_format=torch.contiguous_format) + 1}

    engine = _CompiledTensorTreeScan(
        lambda *args: None, steps=2, compile_options={}, device="cpu", clone_state=clone
    )
    value = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    layout, packs, _ = engine._pack({"x": value}, {}, grad_enabled=True)
    state, _ = layout.unpack(packs)
    assert calls == [None] and state["x"].requires_grad
    torch.testing.assert_close(state["x"], 2 * value + 1)
    (gradient,) = torch.autograd.grad(state["x"], value)
    torch.testing.assert_close(gradient, torch.tensor(2.0, dtype=value.dtype))


@pytest.mark.parametrize("kind", ("subclass", "function_mode", "dispatch_mode"))
def test_scan_host_pack_preserves_authored_clone_dispatch_and_error(kind):
    events = []
    failure = RuntimeError("second scan input clone")
    fail = False
    authored = None

    def record(function, args, kwargs):
        selected = (
            function is torch.ops.aten.clone.default
            if kind == "dispatch_mode"
            else function is torch.Tensor.clone
        )
        if selected and args[0] is authored:
            assert args[1:] == () and kwargs == {
                "memory_format": torch.contiguous_format
            }
            assert torch.is_grad_enabled()
            events.append(id(authored))
            if fail and len(events) == 2:
                raise failure

    value = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    if kind == "subclass":

        class Value(torch.Tensor):
            @classmethod
            def __torch_function__(cls, function, types, args=(), kwargs=None):
                record(function, args, kwargs or {})
                return super().__torch_function__(function, types, args, kwargs or {})

        authored = torch.Tensor._make_subclass(Value, value, True)
        context = nullcontext
    elif kind == "function_mode":

        class Mode(TorchFunctionMode):
            def __torch_function__(self, function, types, args=(), kwargs=None):
                record(function, args, kwargs or {})
                return function(*args, **(kwargs or {}))

        authored, context = value, Mode
    else:

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                record(function, args, kwargs or {})
                return function(*args, **(kwargs or {}))

        authored, context = value, Mode
    engine = _CompiledTensorTreeScan(
        lambda *args: None, steps=2, compile_options={}, device="cpu"
    )
    with context():
        layout, packs, _ = engine._pack(
            {"x": authored, "same": authored}, {}, grad_enabled=True
        )
    state, _ = layout.unpack(packs)
    assert events == [id(authored)] * 2
    (gradient,) = torch.autograd.grad(state["x"] + state["same"], authored)
    torch.testing.assert_close(gradient, torch.full_like(authored, 2))
    events.clear()
    fail = True
    with context(), pytest.raises(RuntimeError) as caught:
        engine._pack({"x": authored, "same": authored}, {}, grad_enabled=True)
    fail = False
    assert caught.value is failure and events == [id(authored)] * 2


def test_restored_unused_parameter_reuses_compilation_and_keeps_suffix_gradients(
    monkeypatch,
):
    def body(parameters, prepared, state, carries, sample):
        # The raw source's policy is captured before unused operand pruning.
        policy = 2.0 if parameters["p"].is_leaf else 11.0
        updated = {"x": state["x"] + policy * prepared["g"] * sample["u"]}
        return (updated, carries), updated

    def bindings(value):
        return {"p": value}, {"g": value.square() + 0.2}

    def reference(value, initial, drive):
        # Independent recurrence for the two raw leaf-source invocations.
        increment = 2.0 * (value.square() + 0.2)
        current, frames = initial, []
        for index in range(2):
            current = current + increment * drive[index]
            frames.append(current)
        return {"x": current}, {}, {"x": torch.stack(frames)}

    def objective(result):
        return result[0]["x"].square() + result[2]["x"].square().sum()

    engine, compilation_count, captures = _engine(monkeypatch, body)
    parameter = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    initial = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    prefix_drive = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    suffix_drive = torch.tensor([-0.2, 0.4], dtype=torch.float64, requires_grad=True)
    parameters, prepared = bindings(parameter)
    before = _snapshot((parameters, prepared, initial, prefix_drive, suffix_drive))
    inputs = _leaves((parameter, initial, prefix_drive, suffix_drive))
    versions = [value._version for value in inputs]
    first = engine(parameters, prepared, {"x": initial}, {}, {"u": prefix_drive})
    backend_count = compilation_count()
    assert backend_count > 0 and len(captures) == len(engine._kernels) == 1
    oracle_parameter = parameter.detach().clone().requires_grad_()
    oracle_initial = initial.detach().clone().requires_grad_()
    oracle_prefix_drive = prefix_drive.detach().clone().requires_grad_()
    oracle_first = reference(oracle_parameter, oracle_initial, oracle_prefix_drive)
    torch.testing.assert_close(first, oracle_first, rtol=2e-11, atol=2e-12)
    gradients = torch.autograd.grad(
        objective(first), (parameter, initial, prefix_drive)
    )
    oracle = torch.autograd.grad(
        objective(oracle_first),
        (oracle_parameter, oracle_initial, oracle_prefix_drive),
    )
    torch.testing.assert_close(gradients, oracle, rtol=2e-10, atol=2e-11)
    assert all(
        torch.isfinite(value).all() and torch.count_nonzero(value) > 0
        for value in gradients
    )
    assert compilation_count() == backend_count

    # Portable restoration supplies new regular leaves with saved trainability.
    restored_parameter = parameter.detach().clone().requires_grad_()
    restored_initial = first[0]["x"].detach().clone().requires_grad_()
    assert type(restored_parameter) is torch.Tensor and restored_parameter.is_leaf
    restored_parameters, restored_prepared = bindings(restored_parameter)
    final = engine(
        restored_parameters,
        restored_prepared,
        {"x": restored_initial},
        {},
        {"u": suffix_drive},
    )
    assert compilation_count() == backend_count
    # The restored Tensor has a distinct raw authored capture, while its
    # identical numerical FX program reuses the compiled Parameter executor.
    assert len(captures) == len(engine._kernels) == 2
    ref_parameter = restored_parameter.detach().clone().requires_grad_()
    ref_initial = restored_initial.detach().clone().requires_grad_()
    ref_drive = suffix_drive.detach().clone().requires_grad_()
    expected = reference(ref_parameter, ref_initial, ref_drive)
    torch.testing.assert_close(final, expected, rtol=2e-11, atol=2e-12)
    observed = torch.autograd.grad(
        objective(final),
        (
            restored_parameter,
            restored_initial,
            suffix_drive,
            parameter,
            initial,
            prefix_drive,
        ),
        allow_unused=True,
    )
    wanted = torch.autograd.grad(
        objective(expected), (ref_parameter, ref_initial, ref_drive)
    )
    torch.testing.assert_close(observed[:3], wanted, rtol=2e-10, atol=2e-11)
    assert observed[3:] == (None, None, None)
    assert all(
        torch.isfinite(value).all() and torch.count_nonzero(value) > 0
        for value in observed[:3]
    )
    assert compilation_count() == backend_count
    assert len(captures) == len(engine._kernels) == 2
    torch.testing.assert_close(
        (parameters, prepared, initial, prefix_drive, suffix_drive),
        before,
        rtol=0,
        atol=0,
    )
    assert [value._version for value in inputs] == versions
    assert parameter.requires_grad and parameter.is_leaf


def test_restored_used_parameter_aliases_reuse_compilation_and_preserve_raw_policy(
    monkeypatch,
):
    def body(parameters, prepared, state, carries, sample):
        policy = 2.0 if parameters["p"].is_leaf else 11.0
        coefficient = parameters["p"] + parameters["alias"] + prepared["same"]
        updated = {"x": state["x"] + policy * coefficient * sample["u"]}
        return ((updated, carries), updated)

    def bindings(value):
        return ({"p": value, "alias": value}, {"same": value})

    def oracle(value, initial, drive, *, leaf=True):
        coefficient = (2.0 if leaf else 11.0) * 3 * value
        (current, frames) = (initial, [])
        for index in range(2):
            current = current + coefficient * drive[index]
            frames.append(current)
        return ({"x": current}, {}, {"x": torch.stack(frames)})

    def objective(result):
        return result[0]["x"].square() + result[2]["x"].square().sum()

    (engine, compilation_count, captures) = _engine(monkeypatch, body)
    parameter = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    initial = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    prefix_drive = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    suffix_drive = torch.tensor([-0.2, 0.4], dtype=torch.float64, requires_grad=True)
    (parameters, prepared) = bindings(parameter)
    inputs = (parameters, prepared, initial, prefix_drive, suffix_drive)
    saved = _snapshot(inputs)
    versions = [value._version for value in _leaves(inputs)]
    first = engine(parameters, prepared, {"x": initial}, {}, {"u": prefix_drive})
    _owned(first, inputs)
    first_backend_count = compilation_count()
    assert first_backend_count > 0 and len(captures) == len(engine._kernels) == 1
    ref_parameter = parameter.detach().clone().requires_grad_()
    ref_initial = initial.detach().clone().requires_grad_()
    ref_drive = prefix_drive.detach().clone().requires_grad_()
    expected_first = oracle(ref_parameter, ref_initial, ref_drive)
    torch.testing.assert_close(first, expected_first, rtol=2e-11, atol=2e-12)
    gradients = torch.autograd.grad(
        objective(first), (parameter, initial, prefix_drive)
    )
    expected_gradients = torch.autograd.grad(
        objective(expected_first), (ref_parameter, ref_initial, ref_drive)
    )
    torch.testing.assert_close(gradients, expected_gradients, rtol=2e-10, atol=2e-11)
    assert all(
        (
            torch.isfinite(value).all() and torch.count_nonzero(value) > 0
            for value in gradients
        )
    )
    restored_parameter = parameter.detach().clone().requires_grad_()
    restored_initial = first[0]["x"].detach().clone().requires_grad_()
    (restored_parameters, restored_prepared) = bindings(restored_parameter)
    assert type(restored_parameter) is torch.Tensor and restored_parameter.is_leaf
    assert parameters["p"] is parameters["alias"] is prepared["same"] is parameter
    assert (
        restored_parameters["p"]
        is restored_parameters["alias"]
        is restored_prepared["same"]
        is restored_parameter
    )
    final = engine(
        restored_parameters,
        restored_prepared,
        {"x": restored_initial},
        {},
        {"u": suffix_drive},
    )
    _owned(final, (inputs, first, restored_parameter, restored_initial))
    restored_backend_count = compilation_count()
    assert restored_backend_count == first_backend_count
    assert len(captures) == len(engine._kernels) == 2
    ref_parameter = restored_parameter.detach().clone().requires_grad_()
    ref_initial = restored_initial.detach().clone().requires_grad_()
    ref_drive = suffix_drive.detach().clone().requires_grad_()
    expected = oracle(ref_parameter, ref_initial, ref_drive)
    torch.testing.assert_close(final, expected, rtol=2e-11, atol=2e-12)
    observed = torch.autograd.grad(
        objective(final),
        (
            restored_parameter,
            restored_initial,
            suffix_drive,
            parameter,
            initial,
            prefix_drive,
        ),
        allow_unused=True,
    )
    wanted = torch.autograd.grad(
        objective(expected), (ref_parameter, ref_initial, ref_drive)
    )
    torch.testing.assert_close(observed[:3], wanted, rtol=2e-10, atol=2e-11)
    assert observed[3:] == (None, None, None)
    assert all(
        (
            torch.isfinite(value).all() and torch.count_nonzero(value) > 0
            for value in observed[:3]
        )
    )
    assert compilation_count() == restored_backend_count
    upstream = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    nonleaf = 1.3 * upstream
    (nonleaf_parameters, nonleaf_prepared) = bindings(nonleaf)
    nonleaf_initial = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    nonleaf_drive = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
    changed = engine(
        nonleaf_parameters,
        nonleaf_prepared,
        {"x": nonleaf_initial},
        {},
        {"u": nonleaf_drive},
    )
    ref_upstream = upstream.detach().clone().requires_grad_()
    ref_initial = nonleaf_initial.detach().clone().requires_grad_()
    ref_drive = nonleaf_drive.detach().clone().requires_grad_()
    expected = oracle(1.3 * ref_upstream, ref_initial, ref_drive, leaf=False)
    torch.testing.assert_close(changed, expected, rtol=2e-11, atol=2e-12)
    observed = torch.autograd.grad(
        objective(changed), (upstream, nonleaf_initial, nonleaf_drive)
    )
    wanted = torch.autograd.grad(
        objective(expected), (ref_upstream, ref_initial, ref_drive)
    )
    torch.testing.assert_close(observed, wanted, rtol=2e-10, atol=2e-11)
    assert len(captures) == len(engine._kernels) == 3
    torch.testing.assert_close(inputs, saved, rtol=0, atol=0)
    assert [value._version for value in _leaves(inputs)] == versions
    assert parameter.requires_grad and parameter.is_leaf


def test_captured_parameter_views_keep_dense_frame_aliases_and_saved_tensor_hooks():
    parameter = torch.nn.Parameter(torch.tensor([0.2, 0.7], dtype=torch.float64))
    ordinary = torch.tensor([0.4, 0.9], dtype=torch.float64, requires_grad=True)
    operands = (
        {"p": parameter, "alias": parameter, "ordinary": ordinary},
        {"same": parameter},
        {"field": parameter},
        ({"u": parameter},),
    )
    usage = pytree.tree_map(lambda value: True, operands)
    isolate = _LiftedViewIsolation.from_usage(usage, captured_inputs=True)
    events = []

    def pack(value):
        events.append("pack")
        return value

    def unpack(value):
        events.append("unpack")
        return value

    before, version = parameter.detach().clone(), parameter._version
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        result = isolate(operands)
        assert events == []  # The alias operation itself saves no tensors.
        first = result[0]["p"]
        assert type(first) is torch.Tensor and first.requires_grad and not first.is_leaf
        assert (
            first
            is result[0]["alias"]
            is result[1]["same"]
            is result[2]["field"]
            is result[3][0]["u"]
        )
        assert result[0]["ordinary"] is ordinary
        repeated = isolate(operands)
        assert repeated[0]["p"] is not first
        assert repeated[0]["p"] is repeated[0]["alias"]
        assert events == []
        loss = first.square().sum() + 2 * result[1]["same"].square().sum()
    (gradient,) = torch.autograd.grad(loss, parameter)
    torch.testing.assert_close(gradient, 6 * parameter)
    assert "pack" in events and "unpack" in events
    assert torch._C._is_alias_of(first, parameter)
    assert operands[0]["p"] is parameter and parameter.is_leaf
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert parameter._version == version


@pytest.mark.parametrize("kind", ("subclass", "function_mode", "dispatch_mode"))
def test_captured_parameter_alias_normalization_respects_subclasses_and_modes(kind):
    if kind == "subclass":

        class Parameter(torch.nn.Parameter):
            pass

        parameter = Parameter(torch.tensor(0.7, dtype=torch.float64))
        context = nullcontext
    else:
        parameter = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        if kind == "function_mode":

            class Mode(TorchFunctionMode):
                def __torch_function__(self, function, types, args=(), kwargs=None):
                    return function(*args, **(kwargs or {}))
        else:

            class Mode(TorchDispatchMode):
                def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                    return function(*args, **(kwargs or {}))

        context = Mode
    operands = {"first": parameter, "second": parameter}
    isolate = _LiftedViewIsolation.from_usage(
        {"first": True, "second": True}, captured_inputs=True
    )
    with context():
        result = isolate(operands)
    assert result is operands
    assert result["first"] is result["second"] is parameter
    assert parameter.requires_grad and parameter.is_leaf


@pytest.mark.parametrize("kind", ("offset", "expanded", "conjugate", "negative"))
def test_captured_parameter_view_formats_and_logical_gradients_remain_exact(kind):
    base = torch.arange(1, 13, dtype=torch.float64)
    if kind == "expanded":
        value = base[3:4].expand(3)
    elif kind == "conjugate":
        value = torch.complex(base, base + 0.5)[3:9:2].conj()
    elif kind == "negative":
        value = torch._neg_view(base[3:9:2])
    else:
        value = base[3:9:2]
    parameter = torch.nn.Parameter(value)
    isolate = _LiftedViewIsolation.from_usage({"p": True}, captured_inputs=True)
    before, version = parameter.detach().clone(), parameter._version
    original_metadata = (
        parameter.shape,
        parameter.stride(),
        parameter.storage_offset(),
        parameter.dtype,
        parameter.device,
        parameter.is_conj(),
        parameter.is_neg(),
    )
    result = isolate({"p": parameter})["p"]
    assert type(result) is torch.Tensor and result.requires_grad
    assert (
        result.shape,
        result.stride(),
        result.storage_offset(),
        result.dtype,
        result.device,
        result.is_conj(),
        result.is_neg(),
    ) == original_metadata
    torch.testing.assert_close(result, parameter, rtol=0, atol=0)
    (gradient,) = torch.autograd.grad(result.abs().square().sum(), parameter)
    torch.testing.assert_close(gradient, 2 * parameter, rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert parameter._version == version and parameter.is_leaf


@pytest.mark.parametrize("owner", ("instance", "parameter_class"))
def test_captured_parameter_clone_hooks_keep_legacy_dense_values_and_vjps(
    monkeypatch, owner
):
    from types import MethodType

    from dendra.func._hop_inputs import _isolate_dense_lifted_inputs

    parameter = torch.nn.Parameter(torch.tensor([0.2, 0.7], dtype=torch.float64))
    operands = {"p": parameter}
    isolate = _LiftedViewIsolation.from_usage({"p": True}, captured_inputs=True)
    original = torch.Tensor.clone
    events = []
    before, version = parameter.detach().clone(), parameter._version

    def clone(value, **kwargs):
        events.append(kwargs)
        return original(value, **kwargs) + 3

    if owner == "instance":
        monkeypatch.setattr(parameter, "clone", MethodType(clone, parameter))
    else:
        monkeypatch.setattr(torch.nn.Parameter, "clone", clone)
    expected = _isolate_dense_lifted_inputs(operands)["p"]
    (expected_gradient,) = torch.autograd.grad(expected.square().sum(), parameter)
    normalized = isolate(operands)
    assert normalized is operands and normalized["p"] is parameter
    actual = _isolate_dense_lifted_inputs(normalized)["p"]
    (gradient,) = torch.autograd.grad(actual.square().sum(), parameter)
    assert events == [{"memory_format": torch.preserve_format}] * 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(actual, parameter + 3)
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert parameter._version == version and parameter.is_leaf
