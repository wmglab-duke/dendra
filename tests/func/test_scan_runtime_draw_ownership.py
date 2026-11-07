"""Runtime draw ownership avoids scan recapture without changing authored inputs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._draws import RuntimeDraws
from dendra.func._hop_inputs import _own_runtime_draw_packets
from dendra.func._inputs import _RuntimeInputs
from dendra.func._netstim import NetStimDraw
from dendra.func._scan import _CompiledScanPopulationChunk
from dendra.func._scan_tree import _CompiledTensorTreeScan
from tests.func.test_network import DT, _stim_network
from tests.func.test_stimulation_rng_sharing import _population

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
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
        torch.set_num_threads(previous)


def _tensors(tree):
    return [value for value in pytree.tree_leaves(tree) if torch.is_tensor(value)]


def _copy(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _slice(tree, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], tree)


def _packet(kind="runtime"):
    value = torch.arange(60, dtype=torch.float64).reshape(10, 2, 3)[2:6]
    rng = torch.arange(80, dtype=torch.uint8).reshape(10, 8)[2:6]
    if kind == "legacy":
        return NetStimDraw(value, rng)
    return RuntimeDraws({"value": value, "alias": value}, {"stream": {"cpu": rng}})


@pytest.mark.parametrize("kind", ("runtime", "legacy"))
def test_plain_offset_packets_own_fresh_storage_without_mutating_sources(kind):
    packet = _packet(kind)
    before = _copy(packet)
    originals = _tensors(packet)
    versions = [value._version for value in originals]
    result = _own_runtime_draw_packets(packet)
    assert type(result) is type(packet) and result is not packet
    torch.testing.assert_close(result, before)
    leaves = _tensors(result)
    assert all(
        value.is_contiguous() and value.storage_offset() == 0 for value in leaves
    )
    for index, value in enumerate(leaves):
        assert all(
            not torch._C._is_alias_of(value, other)
            for other in (*leaves[index + 1 :], *originals)
        )
    torch.testing.assert_close(packet, before)
    assert [value._version for value in originals] == versions


def test_deterministic_input_objects_and_view_metadata_remain_live():
    source = torch.arange(120, dtype=torch.float64).reshape(10, 12)
    ve, intra = source[2:6, ::2], source[3:7, 1::2]
    packet = _packet()
    inputs = _RuntimeInputs(ve=ve, intra=intra, draws=packet)
    tree = {"populations": {"cell": inputs}, "legacy": [dn.func.Inputs(ve, intra)]}
    result = _own_runtime_draw_packets(tree)
    selected = result["populations"]["cell"]
    assert selected.draws is not packet
    assert selected.ve is ve and selected.intra is intra
    assert result["legacy"] is tree["legacy"]
    assert result["legacy"][0] is tree["legacy"][0]
    assert selected.ve.stride() == (12, 2) and selected.ve.storage_offset() == 24
    assert selected.intra.stride() == (12, 2) and selected.intra.storage_offset() == 37


def test_owned_draw_copies_preserve_live_gradients_and_alias_contributions():
    source = torch.arange(16, dtype=torch.float64, requires_grad=True)
    value = source.reshape(8, 2)[2:6]
    packet = RuntimeDraws({"first": value, "second": value}, {})
    owned = _own_runtime_draw_packets(packet)
    loss = owned.draws["first"].square().sum() + 3 * owned.draws["second"].sum()
    observed = torch.autograd.grad(loss, source, retain_graph=True)[0]
    expected = torch.autograd.grad(value.square().sum() + 3 * value.sum(), source)[0]
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert owned.draws["first"].requires_grad
    assert owned.draws["first"].grad_fn is not None
    assert source._version == 0


def test_inference_draw_tapes_become_ordinary_independent_owned_tensors():
    with torch.inference_mode():
        packet = _packet()
        result = _own_runtime_draw_packets(packet)
    assert all(value.is_inference() for value in _tensors(packet))
    assert all(not value.is_inference() for value in _tensors(result))
    torch.testing.assert_close(result, packet)


@pytest.mark.parametrize("layout", ("transpose", "singleton"))
def test_noncanonical_draw_layouts_keep_the_original_packet(layout):
    if layout == "transpose":
        value = torch.arange(24, dtype=torch.float64).reshape(4, 2, 3).transpose(1, 2)
    else:
        value = torch.arange(12, dtype=torch.float64).as_strided((4, 1, 3), (3, 99, 1))
    packet = RuntimeDraws({"value": value}, {})
    assert _own_runtime_draw_packets(packet) is packet


@pytest.mark.parametrize("kind", ("subclass", "function_mode", "dispatch_mode"))
def test_authored_tensor_dispatch_declines_without_invoking_tensor_operations(kind):
    calls = []
    value = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    context = nullcontext
    if kind == "subclass":

        class Value(torch.Tensor):
            @classmethod
            def __torch_function__(cls, function, types, args=(), kwargs=None):
                calls.append(function)
                raise AssertionError("authored Tensor operation")

        value = torch.Tensor._make_subclass(Value, value, False)
    elif kind == "function_mode":

        class Mode(TorchFunctionMode):
            def __torch_function__(self, function, types, args=(), kwargs=None):
                calls.append(function)
                raise AssertionError("authored function mode")

        context = Mode
    else:

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                calls.append(function)
                raise AssertionError("authored dispatch mode")

        context = Mode
    packet = RuntimeDraws({"value": value}, {})
    with context():
        assert _own_runtime_draw_packets(packet) is packet
    assert calls == []


@pytest.mark.parametrize("method", ("stride", "clone"))
@pytest.mark.parametrize("scope", ("class", "instance"))
def test_tensor_method_overrides_decline_before_any_authored_call(
    monkeypatch, method, scope
):
    packet = _packet()
    calls = []

    def observed(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("authored Tensor method")

    owner = torch.Tensor if scope == "class" else packet.draws["value"]
    monkeypatch.setattr(owner, method, observed)
    assert _own_runtime_draw_packets(packet) is packet
    assert calls == []


def test_opaque_packet_mapping_is_not_traversed():
    calls = []

    class Opaque(Mapping):
        def __iter__(self):
            calls.append("iter")
            raise AssertionError("authored Mapping")

        def __getitem__(self, key):
            calls.append(key)
            raise AssertionError("authored Mapping")

        def __len__(self):
            calls.append("len")
            raise AssertionError("authored Mapping")

    packet = RuntimeDraws(Opaque(), {})
    assert _own_runtime_draw_packets(packet) is packet
    assert calls == []


@pytest.mark.parametrize(
    "packet", (RuntimeDraws({}, {}), RuntimeDraws({"none": None}, {}))
)
def test_empty_draw_packets_keep_their_original_identity(packet):
    assert _own_runtime_draw_packets(packet) is packet


@pytest.mark.parametrize("network", [False, True], ids=["population", "network"])
def test_scan_reuses_captures_for_returned_and_portable_draw_continuations(
    monkeypatch, network
):
    if network:
        model = _stim_network()
        generator = model.netstim._rng
        fields = ("cell.v", "t")
        engine = _CompiledTensorTreeScan
    else:
        model, _, pulse = _population()
        generator = pulse.generator
        fields = ("v", "t")
        engine = _CompiledScanPopulationChunk
    model.eval()
    captures = []
    graphs = []
    original = engine._capture

    def backend(graph, _inputs):
        graphs.append(graph.code)
        return graph.forward

    def capture(self, *args, **kwargs):
        captures.append(self._steps)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(engine, "_capture", capture)
    runner = dn.compile(
        model,
        dt=DT,
        steps=9,
        chunk_steps=4,
        execution="scan",
        inference=True,
        backend=backend,
        callbacks={"trace": dn.func.Recorder(fields)},
    )
    initial = _copy(runner.initial_state)
    source = _copy(runner.functional.extract())
    global_rng = torch.get_rng_state().clone()
    native_rng = generator.get_state().clone()
    tape = runner.functional.sample_runtime_draws(initial, steps=18)
    frozen = _copy(tape)
    with torch.no_grad():
        expected = runner.functional.prepare_and_rollout(
            runner.parameters,
            runner.constants,
            initial,
            steps=18,
            draws=tape,
            callbacks=runner.callbacks,
        )
    first = runner(draws=_slice(tape, 0, 9))
    admitted = tuple(captures)
    admitted_graphs = tuple(graphs)
    assert sorted(admitted) == [1, 4]
    assert admitted_graphs
    returned = first[1]["continuation"]
    portable = runner.restore_continuation(runner.export_continuation(returned))
    # Inference-created explicit tapes have the same values and logical schema
    # as ordinary sampled tapes, but must not specialize a second scan kernel.
    with torch.inference_mode():
        suffix_tape = pytree.tree_map_only(
            torch.Tensor, lambda value: value.clone(), _slice(tape, 9, 18)
        )
    assert all(torch.is_inference(value) for value in _tensors(suffix_tape))
    for continuation in (returned, portable):
        actual = runner(continuation, draws=suffix_tape)
        torch.testing.assert_close(actual[0], expected[0], rtol=2e-10, atol=2e-11)
        for name, beginning in first[1]["callbacks"]["trace"].items():
            suffix = actual[1]["callbacks"]["trace"][name]
            assert beginning.shape[0] == 10 and suffix.shape[0] == 9
            torch.testing.assert_close(
                torch.cat((beginning, suffix)),
                expected[1]["callbacks"]["trace"][name],
                rtol=2e-10,
                atol=2e-11,
            )
        assert tuple(captures) == admitted
        assert tuple(graphs) == admitted_graphs
    torch.testing.assert_close(runner.initial_state, initial)
    torch.testing.assert_close(runner.functional.extract(), source)
    torch.testing.assert_close(tape, frozen)
    torch.testing.assert_close(generator.get_state(), native_rng)
    torch.testing.assert_close(torch.get_rng_state(), global_rng)


def _compiler_activity():
    from torch._dynamo import utils as compiler_utils

    return (
        deepcopy(compiler_utils.counters),
        list(compiler_utils.get_compilation_metrics()),
        {
            name: len(values)
            for name, values in compiler_utils.compilation_time_metrics.items()
        },
    )


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"], ids=["aot-eager", "cpp"])
def test_scan_inference_draw_dispatch_keys_do_not_recompile(backend):
    from tests.func.test_runtime_noise import DT as NOISE_DT
    from tests.func.test_runtime_noise import _population as noisy_population

    model = noisy_population()
    model.eval()
    options = (
        {"cpp_wrapper": True, "compile_threads": 1} if backend == "inductor" else None
    )
    runner = dn.compile(
        model,
        dt=NOISE_DT,
        steps=9,
        chunk_steps=8,
        execution="scan",
        inference=True,
        backend=backend,
        options=options,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
    )
    initial = _copy(runner.initial_state)
    native = _copy(runner.functional.extract())
    global_rng = torch.get_rng_state().clone()
    tape = runner.functional.sample_runtime_draws(initial, steps=18)
    sources = (runner.parameters, runner.constants, runner.initial_state, tape)
    frozen = _copy(sources)
    versioned = [value for value in _tensors(sources) if not torch.is_inference(value)]
    versions = tuple(value._version for value in versioned)
    with torch.no_grad():
        expected = runner.functional.prepare_and_rollout(
            runner.parameters,
            runner.constants,
            initial,
            steps=18,
            draws=tape,
            callbacks=runner.callbacks,
        )
    first = runner(draws=_slice(tape, 0, 9))
    returned = first[1]["continuation"]
    portable = runner.restore_continuation(runner.export_continuation(returned))
    with torch.inference_mode():
        suffix = pytree.tree_map_only(
            torch.Tensor, lambda value: value.clone(), _slice(tape, 9, 18)
        )
    suffix_leaves = _tensors(suffix)
    assert suffix_leaves and all(torch.is_inference(value) for value in suffix_leaves)
    before = _compiler_activity()
    assert before[1], "The positive control must compile its scan kernels"
    for continuation in (returned, portable):
        actual = runner(continuation, draws=suffix)
        torch.testing.assert_close(actual[0], expected[0], rtol=2e-10, atol=2e-11)
        for name, prefix in first[1]["callbacks"]["trace"].items():
            tail = actual[1]["callbacks"]["trace"][name]
            assert prefix.shape[0] == 10 and tail.shape[0] == 9
            torch.testing.assert_close(
                torch.cat((prefix, tail)),
                expected[1]["callbacks"]["trace"][name],
                rtol=2e-10,
                atol=2e-11,
            )
        assert _compiler_activity() == before, (
            "Inference draw dispatch keys triggered compilation"
        )
    torch.testing.assert_close(sources, frozen)
    assert tuple(value._version for value in versioned) == versions
    torch.testing.assert_close(suffix, _slice(tape, 9, 18))
    torch.testing.assert_close(runner.functional.extract(), native)
    torch.testing.assert_close(torch.get_rng_state(), global_rng)
