"""Large CPU while bodies use a local, qualified code-generation boundary."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests._call_observer import observe_calls

pytestmark = pytest.mark.cpu


def _module():
    return importlib.import_module("dendra.func._while_outlining")


def _known_support():
    module = _module()
    root = Path(torch.__file__).resolve().parent / "_inductor"
    try:
        known = all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
            for name, expected in module._SOURCES.items()
        )
    except OSError:
        known = False
    support = module._support()
    if not known:
        assert support is None
        pytest.skip("Installed compiler sources do not match the qualified contract")
    assert support is not None, "Recognized compiler sources must qualify"
    assert module._current(support[2])
    return support


def _eligible_options():
    return {
        "backend": "inductor",
        "fullgraph": True,
        "options": {
            "cpp_wrapper": True,
            "cpp_wrapper_build_separate": False,
            "compile_threads": 1,
        },
    }


def _closure(function):
    return dict(
        zip(function.__code__.co_freevars, function.__closure__ or (), strict=True)
    )


def _copy_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {name: _copy_tree(child) for name, child in value.items()}
    if isinstance(value, tuple):
        children = [_copy_tree(child) for child in value]
        return type(value)(*children) if hasattr(value, "_fields") else tuple(children)
    if isinstance(value, list):
        return [_copy_tree(child) for child in value]
    return value


def _tensor_leaves(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensor_leaves(child)


def _assert_tree_close(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    elif isinstance(expected, Mapping):
        assert tuple(actual) == tuple(expected)
        for name, child in expected.items():
            _assert_tree_close(actual[name], child)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for child, reference in zip(actual, expected, strict=True):
            _assert_tree_close(child, reference)
    elif expected is None or type(expected) in (str, int, float, bool):
        assert actual == expected
    else:
        assert actual is expected


@pytest.fixture(autouse=True)
def _warnings():
    with torch_compiler_warning_context():
        yield


def test_qualified_options_copy_only_owned_mapping_and_preserve_global_owners():
    module = _module()
    config, _, _ = _known_support()
    from torch._inductor.codegen import common, cpp_wrapper_cpu

    options = _eligible_options()
    frozen = deepcopy(options)
    registry = common.device_codegens
    registry_items = tuple(registry.items())
    methods = dict(vars(cpp_wrapper_cpu.CppWrapperCpu))
    previous = {
        name: getattr(config, name)
        for name in ("cpp_wrapper", "cpp_wrapper_build_separate", *module._HOOKS)
    }
    result = module._while_outlining_options(options, device=torch.device("cpu"))
    if sys.platform not in ("darwin", "linux"):
        assert result is options
        return
    custom_pass = result["options"]["_post_fusion_custom_pass"]
    assert type(custom_pass) is module._OutlineWhileBodyPass
    assert result is not options and result["options"] is not options["options"]
    assert options == frozen
    assert result == {
        **options,
        "options": {**options["options"], "_post_fusion_custom_pass": custom_pass},
    }
    assert custom_pass.uuid() == custom_pass.uuid()
    assert isinstance(custom_pass.uuid(), bytes) and custom_pass.uuid()
    assert common.device_codegens is registry
    assert tuple(registry.items()) == registry_items
    assert dict(vars(cpp_wrapper_cpu.CppWrapperCpu)) == methods
    assert {name: getattr(config, name) for name in previous} == previous


@pytest.mark.parametrize(
    "update,device",
    [
        ({"backend": "aot_eager"}, "cpu"),
        ({"backend": lambda graph, examples: graph.forward}, "cpu"),
        ({"disable": True}, "cpu"),
        ({"mode": "reduce-overhead"}, "cpu"),
        ({}, "mps"),
        ({}, "cuda"),
    ],
)
def test_unowned_execution_keeps_exact_options_identity(update, device):
    options = {**_eligible_options(), **update}
    result = _module()._while_outlining_options(options, device=torch.device(device))
    assert result is options


@pytest.mark.parametrize(
    "name,value",
    [
        ("cpp_wrapper", False),
        ("cpp_wrapper_build_separate", True),
        ("fx_wrapper", True),
        ("disable_cpp_codegen", True),
        ("memory_planning", True),
        ("cpu_backend", "triton"),
        ("aot_inductor.allow_stack_allocation", True),
        ("aot_inductor.debug_compile", True),
        ("use_aoti", False),
        ("_post_fusion_custom_pass", None),
        ("post_grad_custom_pre_pass", lambda graph: None),
    ],
)
def test_explicit_options_keep_identity_without_qualification(monkeypatch, name, value):
    module = _module()

    def unexpected_support():
        raise AssertionError(
            "Ineligible options must not trigger compiler qualification"
        )

    monkeypatch.setattr(module, "_support", unexpected_support)
    options = _eligible_options()
    options["options"][name] = value
    assert (
        module._while_outlining_options(options, device=torch.device("cpu")) is options
    )


@pytest.mark.parametrize("mutation", ["code", "defaults", "kwdefaults", "descriptor"])
def test_loaded_method_mutations_decline_after_support_cache_is_primed(
    monkeypatch, mutation
):
    module = _module()
    support = _known_support()
    from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu

    function = CppWrapperCpu.codegen_while_loop

    def replacement(self, while_loop, stack_output=False):
        raise AssertionError("Modified compiler body must not run")

    with monkeypatch.context() as patch:
        if mutation == "code":
            patch.setattr(function, "__code__", replacement.__code__)
        elif mutation == "defaults":
            patch.setattr(function, "__defaults__", (True,))
        elif mutation == "kwdefaults":
            patch.setattr(function, "__kwdefaults__", {"authored": object()})
        else:
            patch.setattr(CppWrapperCpu, "codegen_while_loop", replacement)
        assert module._support() is support
        assert not module._current(support[2])
        assert module._OutlineWhileBodyPass().uuid() is None
        options = _eligible_options()
        assert (
            module._while_outlining_options(options, device=torch.device("cpu"))
            is options
        )
    assert module._current(support[2])


def test_unknown_source_declines_initial_qualification_without_caching_a_live_owner(
    monkeypatch,
):
    module = _module()
    _known_support()

    name = next(iter(module._SOURCES))
    try:
        with monkeypatch.context() as patch:
            patch.setitem(module._SOURCES, name, "0" * 64)
            module._support.cache_clear()
            assert module._support() is None
            options = _eligible_options()
            assert (
                module._while_outlining_options(options, device=torch.device("cpu"))
                is options
            )
    finally:
        module._support.cache_clear()
    _known_support()


def test_stateless_pass_instances_have_stable_actual_fx_cache_pickle_bytes():
    _known_support()
    custom_pass = _module()._OutlineWhileBodyPass()
    from torch._inductor.codecache import FxGraphCachePickler

    graph = torch.fx.Graph()
    argument = graph.placeholder("value")
    graph.output(argument)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    other = type(custom_pass)()
    assert other is not custom_pass and other.uuid() == custom_pass.uuid()
    pickler = FxGraphCachePickler(gm)
    first = pickler.dumps({"_post_fusion_custom_pass": custom_pass})
    assert first and first == pickler.dumps({"_post_fusion_custom_pass": other})


@pytest.mark.parametrize("fail", [False, True])
def test_actual_body_adapter_preserves_both_declaration_scopes_and_exceptions(fail):
    module = _module()
    _, apply_pass, _ = _known_support()
    from torch._inductor.codegen.common import IndentedBuffer
    from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
    from torch._inductor.codegen.wrapper import (
        EnterKernelProfileScopeLine,
        ExitKernelProfileScopeLine,
    )

    function = _closure(apply_pass)["codegen_subgraph"].cell_contents
    body = object()
    original_error = RuntimeError("Original body failure")

    class Owner:
        def __init__(self):
            self.lines = []
            self.computed_sizes = {"outer": True}
            self.computed_sizes_stack = []
            self.codegen_int_array_var_cache = {"outer": "outer-array"}
            self._kernel_profile_scope_state = []
            self.counter = 0

        def push_computed_sizes(self, values):
            self.computed_sizes_stack.append(deepcopy(values))

        def pop_computed_sizes(self):
            return self.computed_sizes_stack.pop()

        def writeline(self, value):
            self.lines.append(value)

        def original(self, subgraph, inputs, outputs):
            assert subgraph is body and inputs is operands and outputs is operands
            assert "inner" not in self.codegen_int_array_var_cache
            self.counter += 1
            self.codegen_int_array_var_cache["inner"] = "array_" + str(self.counter)
            self.computed_sizes["inner"] = True
            self.writeline("declare " + self.codegen_int_array_var_cache["inner"])
            if fail:
                raise original_error
            return marker

    owner = Owner()
    owner.push_kernel_profile_scope_state = MethodType(
        CppWrapperCpu.push_kernel_profile_scope_state, owner
    )
    owner.pop_kernel_profile_scope_state = MethodType(
        CppWrapperCpu.pop_kernel_profile_scope_state, owner
    )
    operands, marker = (), object()
    setattr(
        owner,
        module._STATE,
        {"subgraph": owner.original, "bodies": [body], "next": 0},
    )
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError) as caught:
                function(owner, body, operands, operands)
            assert caught.value is original_error
        else:
            assert function(owner, body, operands, operands) is marker
        assert owner.codegen_int_array_var_cache == {"outer": "outer-array"}
        assert owner.computed_sizes == {"outer": True}
        assert not owner.computed_sizes_stack and not owner._kernel_profile_scope_state
    assert owner.counter == 2
    assert (
        sum(isinstance(line, EnterKernelProfileScopeLine) for line in owner.lines) == 2
    )
    assert (
        sum(isinstance(line, ExitKernelProfileScopeLine) for line in owner.lines) == 2
    )
    code = IndentedBuffer()
    for line in owner.lines:
        if type(line) is str:
            code.writeline(line)
        else:
            line.codegen(code)
    assert code.getvalue().count("__attribute__((noinline))") == 2
    assert code.getvalue().count("dendra_while_body_") == 4
    assert not owner._kernel_profile_scope_state
    assert owner.codegen_int_array_var_cache == {"outer": "outer-array"}


def _run_inductor_case(network, reader):
    """Fresh child: exactly one four-step compilation per model, then resume."""
    from torch._dynamo import utils as compiler_utils
    from torch._inductor.codecache import CppWrapperCodeCache

    from dendra.func._tensor_tree_while import _CompiledTensorTreeWhile
    from tests.func.test_poisson_stimulation import DT
    from tests.func.test_stimulation_rng_sharing import _network, _population

    torch.set_num_threads(1)
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        _, apply_pass, _ = _known_support()
        body_adapter = _closure(apply_pass)["codegen_subgraph"].cell_contents
        enter_body = _closure(body_adapter)["EnterBody"].cell_contents
        if network:
            model, pulses = _network()
            fields = ("left.v", "right.v", "t")
        else:
            model, _, pulse = _population()
            pulses = [pulse]
            fields = ("v", "t")
        model.eval()
        runner = dn.compile(
            model,
            dt=DT,
            steps=4,
            chunk_steps=4,
            execution="while",
            inference=True,
            callbacks={"trace": dn.func.Recorder(fields)},
            backend="inductor",
            options={"compile_threads": 1},
        )
        generators = {id(pulse.generator): pulse.generator for pulse in pulses}
        rngs = {key: value.get_state().clone() for key, value in generators.items()}
        global_rng = torch.get_rng_state().clone()
        native = _copy_tree(runner.functional.extract())
        tape = runner.functional.sample_runtime_draws(runner.initial_state, steps=8)
        from torch.utils import _pytree as pytree

        def sliced(start, stop):
            return pytree.tree_map_only(
                torch.Tensor, lambda value: value[start:stop], tape
            )

        sources = (runner.parameters, runner.constants, runner.initial_state, tape)
        frozen = _copy_tree(sources)
        versioned = [
            value for value in _tensor_leaves(sources) if not torch.is_inference(value)
        ]
        versions = [value._version for value in versioned]
        with torch.no_grad():
            eager_prefix = runner.functional.prepare_and_rollout(
                runner.parameters,
                runner.constants,
                runner.initial_state,
                steps=4,
                draws=sliced(0, 4),
                callbacks=runner.callbacks,
            )
            eager = runner.functional.prepare_and_rollout(
                runner.parameters,
                runner.constants,
                runner.initial_state,
                steps=8,
                draws=tape,
                callbacks=runner.callbacks,
            )
        entries = []
        admissions = []
        original_loader = CppWrapperCodeCache.load_pybinding

        def checked_loader(cls, *arguments, **keywords):
            code = keywords.get("main_code")
            assert type(code) is str
            if "while (1)" in code:
                # Fail before native compilation if a guard unexpectedly declines
                # this known large fixture, rather than compiling the old giant loop.
                assert "dendra_while_body_" in code
                assert "__attribute__((noinline))" in code
                admissions.append(True)
            return original_loader(*arguments, **keywords)

        def observed(frame):
            entries.append(frame.f_code)

        before_fx_hits = compiler_utils.counters["inductor"]["fxgraph_cache_hit"]
        before_aot_hits = compiler_utils.counters["aot_autograd"]["autograd_cache_hit"]
        with (
            patch.object(
                CppWrapperCodeCache, "load_pybinding", classmethod(checked_loader)
            ),
            observe_calls(
                {
                    _CompiledTensorTreeWhile._capture.__code__,
                    enter_body.codegen.__code__,
                },
                observed,
            ),
        ):
            prefix = runner(draws=sliced(0, 4))
            _assert_tree_close(prefix[0], eager_prefix[0])
            _assert_tree_close(prefix[1]["callbacks"], eager_prefix[1]["callbacks"])
            first_entries = list(entries)
            assert first_entries.count(_CompiledTensorTreeWhile._capture.__code__) == 1
            assert admissions
            if reader:
                assert first_entries.count(enter_body.codegen.__code__) == 0
                assert (
                    compiler_utils.counters["inductor"]["fxgraph_cache_hit"]
                    > before_fx_hits
                )
            else:
                assert first_entries.count(enter_body.codegen.__code__) >= 1
            counters = deepcopy(compiler_utils.counters)
            metrics = list(compiler_utils.get_compilation_metrics())
            returned = prefix[1]["continuation"]
            restored = runner.restore_continuation(runner.export_continuation(returned))
            for continuation in (returned, restored):
                resumed = runner(continuation, draws=sliced(4, 8))
                _assert_tree_close(resumed[0], eager[0])
                for name, beginning in prefix[1]["callbacks"]["trace"].items():
                    suffix = resumed[1]["callbacks"]["trace"][name]
                    assert beginning.shape[0] == 5 and suffix.shape[0] == 4
                    _assert_tree_close(
                        torch.cat((beginning, suffix)),
                        eager[1]["callbacks"]["trace"][name],
                    )
            assert entries == first_entries
            assert compiler_utils.counters == counters
            assert list(compiler_utils.get_compilation_metrics()) == metrics
        _assert_tree_close(sources, frozen)
        assert [value._version for value in versioned] == versions
        _assert_tree_close(runner.functional.extract(), native)
        for key, generator in generators.items():
            torch.testing.assert_close(generator.get_state(), rngs[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
        assert _known_support()[0]._post_fusion_custom_pass is None
    return {
        "qualified": True,
        "network": network,
        "captures": 1,
        "portable": True,
        "reader": reader,
        "fx_hits": compiler_utils.counters["inductor"]["fxgraph_cache_hit"]
        - before_fx_hits,
        "aot_hits": compiler_utils.counters["aot_autograd"]["autograd_cache_hit"]
        - before_aot_hits,
    }


@pytest.mark.parametrize("network", [False, True], ids=["population", "network"])
def test_inductor_large_noisy_while_default_matches_eager_and_portable_resume(
    tmp_path, network
):
    _known_support()
    if sys.platform not in ("darwin", "linux"):
        options = _eligible_options()
        assert (
            _module()._while_outlining_options(options, device=torch.device("cpu"))
            is options
        )
        return
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCHINDUCTOR_CACHE_DIR=os.fspath(tmp_path / "inductor"),
    )
    script = (
        "import json, runpy, sys; "
        "result = runpy.run_path(sys.argv[1])['_run_inductor_case'](sys.argv[2] == 'network', sys.argv[3] == 'reader'); "
        "print('DENDRA_WHILE_OUTLINE=' + json.dumps(result))"
    )
    for mode in ("writer", "reader"):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                script,
                __file__,
                "network" if network else "population",
                mode,
            ],
            cwd=Path(dn.__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        rows = [
            json.loads(line.removeprefix("DENDRA_WHILE_OUTLINE="))
            for line in completed.stdout.splitlines()
            if line.startswith("DENDRA_WHILE_OUTLINE=")
        ]
        assert len(rows) == 1
        row = rows[0]
        assert row["qualified"] and row["network"] is network
        assert row["captures"] == 1 and row["portable"]
        assert row["reader"] is (mode == "reader")
        if mode == "reader":
            assert row["fx_hits"] >= 1


@pytest.mark.parametrize("fail", [False, True])
def test_lazy_compiler_import_warning_scope_runs_under_the_compiler_lock(
    monkeypatch, fail
):
    module = _module()
    _known_support()
    from contextlib import contextmanager

    from torch._dynamo import convert_frame

    from dendra import _bootstrap

    events = []
    active = []
    original_warning_scope = _bootstrap.torch_compiler_warning_context

    class CompilerLock:
        def __enter__(self):
            assert not active
            active.append(True)
            events.append("lock-enter")

        def __exit__(self, *exception):
            assert active == [True]
            active.pop()
            events.append("lock-exit")

    @contextmanager
    def warning_scope():
        assert active == [True]
        events.append("warning-enter")
        try:
            with original_warning_scope():
                if fail:
                    raise RuntimeError("Compiler import warning-scope failure")
                yield
        finally:
            assert active == [True]
            events.append("warning-exit")

    monkeypatch.setattr(convert_frame, "compile_lock", CompilerLock())
    monkeypatch.setattr(_bootstrap, "torch_compiler_warning_context", warning_scope)
    support = module._support.__wrapped__()
    assert (support is None) is fail
    assert events == ["lock-enter", "warning-enter", "warning-exit", "lock-exit"]
    assert active == []


@pytest.mark.parametrize("compiler_lock", [None, object()])
def test_unknown_compiler_lock_declines_before_warning_scope(
    monkeypatch, compiler_lock
):
    module = _module()
    _known_support()
    from torch._dynamo import convert_frame

    from dendra import _bootstrap

    calls = []

    def unexpected_warning_scope():
        calls.append(True)
        raise AssertionError("An unknown compiler lock must not mutate warning filters")

    monkeypatch.setattr(convert_frame, "compile_lock", compiler_lock)
    monkeypatch.setattr(
        _bootstrap, "torch_compiler_warning_context", unexpected_warning_scope
    )
    assert module._support.__wrapped__() is None
    assert calls == []
