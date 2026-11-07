"""Ordinary while compilation preserves options, full traces and ownership."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from test_recorder_while_loop import _case, _clone, _close
from test_while_block_ownership import _offset_view

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._while_inductor import _compile_while

pytestmark = pytest.mark.cpu


@pytest.fixture(params=[1, 4], ids=["t1", "t4"])
def while_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _compile_stub(monkeypatch):
    captured = {}

    def compiled(*_args):
        return None

    def compile_stub(function, **options):
        captured.update(function=function, options=options)
        return compiled

    monkeypatch.setattr(torch, "compile", compile_stub)
    return captured, compiled


@pytest.mark.parametrize("version", ["2.14.0", "2.14.1", "2.14.2", "2.15.0"])
@pytest.mark.parametrize("backend", [None, "inductor", "eager", "aot_eager"])
def test_while_options_and_patch_versions_keep_ordinary_compile(
    monkeypatch, version, backend
):
    captured, compiled = _compile_stub(monkeypatch)
    monkeypatch.setattr(torch, "__version__", version)

    def prior_hook(nodes):
        return nodes

    supplied = {
        "fullgraph": True,
        "dynamic": False,
        "options": {
            "cpp-wrapper": True,
            "compile-threads": 1,
            "cpp.min-chunk-size": 512,
            "_post_fusion_custom_pass": prior_hook,
        },
    }
    if backend is not None:
        supplied["backend"] = backend
    before = deepcopy(supplied)
    assert _compile_while(lambda value: value, supplied) is compiled
    assert captured["options"] == before and supplied == before
    assert captured["options"]["options"] is not supplied["options"]
    assert not hasattr(compiled, "_dendra_while_memory_diagnostics")
    assert not hasattr(compiled, "_dendra_copy_scheduling_diagnostics")
    assert not hasattr(compiled, "_dendra_scan_donation_diagnostics")


@pytest.mark.parametrize(
    "extra",
    [
        {"backend": "eager"},
        {"backend": "aot_eager"},
        {"backend": lambda graph, inputs: graph.forward},
        {"backend": "inductor", "disable": True},
        {"backend": "inductor", "options": {"use_aoti": True}},
    ],
)
def test_non_inductor_disabled_and_aoti_routes_are_passed_through(monkeypatch, extra):
    captured, compiled = _compile_stub(monkeypatch)
    supplied = {"fullgraph": True, **extra}
    before = deepcopy(supplied)
    assert _compile_while(lambda value: value, supplied) is compiled
    assert captured["options"] == before and supplied == before
    assert not hasattr(compiled, "_dendra_while_memory_diagnostics")


def test_mode_options_conflict_preserves_upstream_error():
    with pytest.raises(RuntimeError, match="Either mode or options"):
        _compile_while(
            lambda value: value,
            {"backend": "inductor", "mode": "default", "options": {}},
        )


def test_callback_capture_failure_declines_without_compiling_unrolled_fallback(
    monkeypatch,
):
    from dendra.func._population_while import _PopulationWhileExecutor

    executor = _PopulationWhileExecutor(
        object(),
        steps=8,
        plans=[SimpleNamespace(callback=object())],
        compile_options={"backend": "inductor", "options": {"cpp_wrapper": True}},
    )
    captures = []

    def fail_capture(*operands):
        captures.append(operands)
        raise RuntimeError("sentinel authored callback capture")

    monkeypatch.setattr(executor, "_capture", fail_capture)
    kernel, reason = executor.kernel_for(None, None, None, None, None, None)
    assert kernel is None and "sentinel authored callback capture" in reason
    assert executor._kernels == {} and len(executor._capture_errors) == 1
    assert executor.kernel_for(None, None, None, None, None, None) == (None, reason)
    assert len(captures) == 1


def _inputs(case, steps=8):
    size = steps * case.model.v.numel()
    return dn.func.RolloutInput(
        ve=_offset_view(
            torch.linspace(-0.05, 0.05, size, dtype=case.model.dtype()).reshape(
                steps, *case.model.shape
            )
        ),
        intra=_offset_view(
            torch.linspace(0.0, 1e-9, size, dtype=case.model.dtype()).reshape(
                steps, *case.model.shape
            )
        ),
    )


def _callbacks(case):
    return case.functional.make_callbacks(
        {
            "trace": dn.func.Recorder(("v", "hh.m", "hh.h", "hh.n", "t")),
            "selected": dn.func.Recorder(
                ("v",), node_indices=[0, case.model.shape[-1] - 1]
            ),
            "anomaly": dn.func.AnomalyDetector(),
            "count": dn.func.APCount(
                threshold=-61.0, node_check=[0, case.model.shape[-1] - 1]
            ),
        }
    )


def _assert_upstream_compilers(kernel, *, recorded):
    functions = []
    if recorded:
        for chunks in kernel._callback_chunks.values():
            for chunk in chunks.values():
                executor = chunk._callback_while
                if executor is not None:
                    functions.extend(executor._kernels.values())
    else:
        functions.append(kernel._compiled)
    assert functions
    for function in functions:
        assert not hasattr(function, "_dendra_scan_donation_diagnostics")
        assert not hasattr(function, "_dendra_while_memory_diagnostics")
        assert not hasattr(function, "_dendra_copy_scheduling_diagnostics")


@pytest.mark.parametrize(
    "route", ["recorded", "recorded_tail", "host_spans", "resumed", "final_only"]
)
@pytest.mark.slow
def test_public_cpp_while_full_traces_callbacks_and_final_state_are_immutable(
    route, while_threads
):
    assert torch.get_num_threads() == while_threads
    case = _case()
    case.tensors.state["integrator"]["v"] = _offset_view(
        case.tensors.state["integrator"]["v"]
    )
    steps = 7 if route == "recorded_tail" else 8
    inputs = _inputs(case, steps)
    assert case.tensors.state["integrator"]["v"].storage_offset() > 0
    assert inputs.ve.storage_offset() > 0 and inputs.intra.storage_offset() > 0
    callbacks = None if route == "final_only" else _callbacks(case)
    original = _clone(
        (case.parameters, case.tensors.constants, case.tensors.state, inputs)
    )
    input_leaves = [
        leaf
        for leaf in torch.utils._pytree.tree_leaves(
            (case.parameters, case.tensors.constants, case.tensors.state, inputs)
        )
        if torch.is_tensor(leaf)
    ]
    versions = tuple(leaf._version for leaf in input_leaves)
    source_voltage = case.model.v.detach().clone()
    kernel = case.functional.compile_rollout_chunk(
        4 if route == "resumed" else steps,
        execution="default",
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={"cpp_wrapper": True, "compile_threads": 1, "fx_graph_cache": False},
    )
    eager = case.functional.bind(case.parameters, case.prepared)
    bound = kernel.bind(case.parameters, case.prepared)
    with torch.inference_mode(), torch_compiler_warning_context():
        expected = dn.func.run(
            eager, case.tensors.state, steps=steps, inputs=inputs, callbacks=callbacks
        )
        if route == "resumed":
            middle, first = dn.func.run(
                bound,
                case.tensors.state,
                steps=4,
                inputs=dn.func.RolloutInput(ve=inputs.ve[:4], intra=inputs.intra[:4]),
                callbacks=callbacks,
            )
            final, auxiliary = dn.func.run(
                bound,
                middle,
                steps=4,
                inputs=dn.func.RolloutInput(ve=inputs.ve[4:], intra=inputs.intra[4:]),
                callbacks=callbacks,
                callback_state=first["callbacks"].state,
            )
            results = dict(auxiliary["callbacks"])
            for name in ("trace", "selected"):
                results[name] = {
                    key: torch.cat(
                        (first["callbacks"][name][key], results[name][key]), dim=0
                    )
                    for key in results[name]
                }
            actual = final, results
        else:
            final, auxiliary = dn.func.run(
                bound,
                case.tensors.state,
                steps=steps,
                inputs=inputs,
                callbacks=callbacks,
                **({"host_span_steps": 4} if route == "host_spans" else {}),
            )
            actual = final, None if callbacks is None else dict(auxiliary["callbacks"])
    assert kernel.execution_report().strategy == "compiled_while_loop"
    _close(actual[0], expected[0])
    assert actual[0]["integrator"]["v"].storage_offset() == 0
    assert (
        actual[0]["integrator"]["v"].stride()
        == case.tensors.state["integrator"]["v"]
        .clone(memory_format=torch.contiguous_format)
        .stride()
    )
    torch.testing.assert_close(auxiliary["v"], expected[1]["v"])
    if callbacks is not None:
        _close(actual[1], dict(expected[1]["callbacks"]))
        for name in ("v", "hh.m", "hh.h", "hh.n", "t"):
            assert actual[1]["trace"][name].shape[0] == steps + 1
            torch.testing.assert_close(
                actual[1]["trace"][name][0], expected[1]["callbacks"]["trace"][name][0]
            )
            torch.testing.assert_close(
                actual[1]["trace"][name][-1],
                expected[1]["callbacks"]["trace"][name][-1],
            )
        assert actual[1]["selected"]["v"].shape[-1] == 2
    _close(
        (case.parameters, case.tensors.constants, case.tensors.state, inputs),
        original,
        exact=True,
    )
    for leaf, version in zip(input_leaves, versions, strict=True):
        assert leaf._version == version
    torch.testing.assert_close(case.model.v, source_voltage, rtol=0, atol=0)
    _assert_upstream_compilers(kernel, recorded=callbacks is not None)


def _while_program(kind, *, steps=6):
    def program(initial, samples, gain):
        # This buffer belongs to this invocation; only its private indexed
        # writes are captured by the loop. No input or persistent buffer is
        # donated/mutated by the authored recurrence.
        frames = torch.empty(
            (steps, *initial.shape), dtype=initial.dtype, device=initial.device
        )

        def apply(index, current):
            sample = torch.index_select(samples, 0, index.reshape(1)).squeeze(0)
            private = current.view_as(current).clone()
            private.add_(sample * gain)
            value = private.tanh()
            frames.index_copy_(0, index.reshape(1), value.unsqueeze(0))
            return value.clone()

        def body(index, current):
            return index + 1, apply(index, current)

        zero = torch.zeros((), dtype=torch.int64, device=initial.device)
        if kind == "nested":

            def outer_condition(index, current):
                return index < steps // 2

            def outer_body(index, current):
                def inner_condition(offset, value):
                    return offset < 2

                def inner_body(offset, value):
                    return offset + 1, apply(index * 2 + offset, value)

                _offset, final = torch.while_loop(
                    inner_condition, inner_body, (zero.clone(), current.clone())
                )
                return index + 1, final.clone()

            _index, final = torch.while_loop(
                outer_condition, outer_body, (zero, initial.clone())
            )
        elif kind == "reused":

            def first_condition(index, current):
                return index < steps // 2

            def last_condition(index, current):
                return index < steps

            index, middle = torch.while_loop(
                first_condition, body, (zero, initial.clone())
            )
            _index, final = torch.while_loop(
                last_condition, body, (index.clone(), middle.clone())
            )
        else:

            def condition(index, current):
                return index < steps

            _index, final = torch.while_loop(condition, body, (zero, initial.clone()))
        # Match Dendra's Recorder contract: return the private mutated buffer
        # directly, then assemble the initial boundary outside this compiled
        # callable. Stock Torch2.14.1 can otherwise constant-fold a cat of
        # empty-derived captured storage to NaNs despite the intervening loop
        # writes (independently reproduced without Dendra's memory repair).
        return final, frames

    return program


def _reference(initial, samples, gain):
    value, frames = initial, [initial]
    for sample in samples.unbind():
        value = (value + sample * gain).tanh()
        frames.append(value)
    return value, torch.stack(frames)


@pytest.mark.parametrize("kind", ["plain", "reused", "nested"])
@pytest.mark.slow
def test_real_cpp_while_private_recording_full_frames_and_immutable_inputs(
    kind, while_threads
):
    assert torch.get_num_threads() == while_threads
    compiled = _compile_while(
        _while_program(kind),
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {
                "cpp_wrapper": True,
                "compile_threads": 1,
                "fx_graph_cache": False,
            },
        },
    )
    first_output, saved_output = None, None
    first_private_frames, saved_private_frames = None, None
    for offset in (0.0, 0.03):
        values = (
            torch.linspace(-0.3, 0.2, 8, dtype=torch.float64) + offset,
            torch.linspace(-0.2, 0.25, 48, dtype=torch.float64).reshape(6, 8),
            torch.tensor(0.2 + offset, dtype=torch.float64),
        )
        snapshots = tuple(value.clone() for value in values)
        versions = tuple(value._version for value in values)
        with torch.inference_mode(), torch_compiler_warning_context():
            raw, expected = compiled(*values), _reference(*values)
            output = raw[0], torch.cat((values[0].unsqueeze(0), raw[1]))
        assert raw[1].shape == (6, 8)
        assert output[1].shape == (7, 8)
        for actual, reference in zip(output, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=2e-11, atol=2e-12)
        for current, saved, version in zip(values, snapshots, versions, strict=True):
            torch.testing.assert_close(current, saved, rtol=0, atol=0)
            assert current._version == version
        assert not hasattr(compiled, "_dendra_while_memory_diagnostics")
        assert not hasattr(compiled, "_dendra_copy_scheduling_diagnostics")
        assert not hasattr(compiled, "_dendra_scan_donation_diagnostics")
        if first_output is None:
            first_output = output
            saved_output = tuple(value.clone() for value in output)
            first_private_frames = raw[1]
            saved_private_frames = raw[1].clone()
        else:
            for actual, reference in zip(first_output, saved_output, strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(
                first_private_frames, saved_private_frames, rtol=0, atol=0
            )
            assert not torch._C._is_alias_of(first_private_frames, raw[1])
