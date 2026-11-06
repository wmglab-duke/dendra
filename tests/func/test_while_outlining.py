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
@pytest.mark.parametrize(
    "active_index", [0, 1], ids=["first-active-subgraph", "last-active-subgraph"]
)
def test_actual_body_adapter_preserves_both_declaration_scopes_and_exceptions(
    fail, active_index
):
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

    class ScalarLine:
        def __init__(self, owner):
            self.owner = owner

        def codegen(self, code):
            assert self.owner.unbacked_symbol_decls == {"outer-scalar"}
            self.owner.unbacked_symbol_decls.add("inner-scalar")
            code.writeline("declare scalar")

    class Owner:
        def __init__(self):
            self.lines = []
            self.computed_sizes = {"outer": True}
            self.computed_sizes_stack = []
            self.codegen_int_array_var_cache = {"outer": "outer-array"}
            self._kernel_profile_scope_state = []
            self.unbacked_symbol_decls = {"outer-scalar"}
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
            assert self.unbacked_symbol_decls == {"outer-scalar"}
            self.unbacked_symbol_decls.add("inner-scalar")
            self.writeline(ScalarLine(self))
            self.counter += 1
            self.codegen_int_array_var_cache["inner"] = "array_" + str(self.counter)
            self.computed_sizes["inner"] = True
            self.writeline("declare " + self.codegen_int_array_var_cache["inner"])
            if fail:
                raise original_error
            return marker

    owner = Owner()
    outer_scalar_decls = owner.unbacked_symbol_decls
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
        {
            "subgraph": owner.original,
            # A selected condition and body share one active scope inventory.
            # Exercise both positions without changing the rollback assertions.
            "bodies": [body, object()] if active_index == 0 else [object(), body],
            "next": 0,
        },
    )
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError) as caught:
                function(owner, body, operands, operands)
            assert caught.value is original_error
        else:
            assert function(owner, body, operands, operands) is marker
        assert owner.codegen_int_array_var_cache == {"outer": "outer-array"}
        assert owner.unbacked_symbol_decls is outer_scalar_decls
        assert owner.unbacked_symbol_decls == {"outer-scalar"}
        assert not getattr(owner, module._STATE)["scalar_decls"]
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
    assert owner.unbacked_symbol_decls is outer_scalar_decls
    assert owner.unbacked_symbol_decls == {"outer-scalar"}
    assert not getattr(owner, module._STATE)["scalar_decls"]


def _run_inductor_case(network, reader, execution="while"):
    """Fresh child: exactly one four-step compilation per model, then resume."""
    from torch._dynamo import utils as compiler_utils
    from torch._inductor.codecache import CppWrapperCodeCache
    from torch._inductor.cpp_builder import CppBuilder

    from dendra.func._tensor_tree_while import _CompiledTensorTreeWhile
    from tests.func.test_poisson_stimulation import DT
    from tests.func.test_stimulation_rng_sharing import _network, _population

    capture_code = _CompiledTensorTreeWhile._capture.__code__
    if execution == "scan":
        from dendra.func._scan import _CompiledScanPopulationChunk
        from dendra.func._scan_tree import _CompiledTensorTreeScan

        capture_code = (
            _CompiledTensorTreeScan if network else _CompiledScanPopulationChunk
        )._capture.__code__
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
            execution=execution,
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
        native_builds = []
        native_loads = {}
        cache_root = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).resolve()

        def native_wrappers():
            return {
                path.relative_to(cache_root).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(cache_root.rglob("*.main.so"))
                if path.is_file()
                and not any(component.endswith(".dSYM") for component in path.parts)
            }

        before_native = native_wrappers() if execution == "scan" else {}
        native_load_code = CppWrapperCodeCache._load_library_inner.__func__.__code__
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
            if execution == "scan" and frame.f_code is CppBuilder.build.__code__:
                sources = frame.f_locals["self"]._sources_args
                if sources.endswith(".main.cpp"):
                    native_builds.append(sources)
                    assert not reader, (
                        "Reader attempted to rebuild an existing native wrapper"
                    )
                return
            if execution == "scan" and frame.f_code is native_load_code:
                path = Path(frame.f_locals["path"]).resolve()
                if path.name.endswith(".main.so"):
                    name = path.relative_to(cache_root).as_posix()
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if reader:
                        assert before_native.get(name) == digest
                    native_loads[name] = digest
                return
            entries.append(frame.f_code)

        before_fx_hits = compiler_utils.counters["inductor"]["fxgraph_cache_hit"]
        before_aot_hits = compiler_utils.counters["aot_autograd"]["autograd_cache_hit"]
        before_fx_bypasses = compiler_utils.counters["inductor"]["fxgraph_cache_bypass"]
        before_aot_bypasses = compiler_utils.counters["aot_autograd"][
            "autograd_cache_bypass"
        ]
        observed_codes = {capture_code, enter_body.codegen.__code__}
        if execution == "scan":
            observed_codes.update((CppBuilder.build.__code__, native_load_code))
        with (
            patch.object(
                CppWrapperCodeCache, "load_pybinding", classmethod(checked_loader)
            ),
            observe_calls(
                observed_codes,
                observed,
            ),
        ):
            prefix = runner(draws=sliced(0, 4))
            _assert_tree_close(prefix[0], eager_prefix[0])
            _assert_tree_close(prefix[1]["callbacks"], eager_prefix[1]["callbacks"])
            first_entries = list(entries)
            assert first_entries.count(capture_code) == 1
            assert admissions
            if reader and execution == "while":
                assert first_entries.count(enter_body.codegen.__code__) == 0
                assert (
                    compiler_utils.counters["inductor"]["fxgraph_cache_hit"]
                    > before_fx_hits
                )
            elif not reader:
                assert first_entries.count(enter_body.codegen.__code__) >= 1
            if execution == "scan":
                assert native_loads
                if reader:
                    assert before_native and not native_builds
                    assert native_wrappers() == before_native
                else:
                    assert native_builds
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
        "fx_bypasses": compiler_utils.counters["inductor"]["fxgraph_cache_bypass"]
        - before_fx_bypasses,
        "aot_bypasses": compiler_utils.counters["aot_autograd"]["autograd_cache_bypass"]
        - before_aot_bypasses,
        "native_wrapper_builds": native_builds,
        "native_wrapper_loads": native_loads,
        "native_wrappers": native_wrappers() if execution == "scan" else {},
        "body_renders": first_entries.count(enter_body.codegen.__code__),
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
        # Cold Network compilation needs headroom on shared CI runners. Keep
        # the cache-reading process and local runs under their existing limit.
        timeout = 600 if os.environ.get("CI") and network and mode == "writer" else 300
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
            timeout=timeout,
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


def _owned_scan_patches():
    from dendra.func._scan_donation import _ScanDonationPass
    from dendra.func._scan_inductor import _CopyViewLowering

    return {
        **_eligible_options()["options"],
        "_pre_fusion_custom_pass": _ScanDonationPass(),
        "post_grad_custom_post_pass": (_CopyViewLowering(),),
    }


def test_scan_outlining_preserves_owned_hook_objects_order_and_global_configuration():
    module = _module()
    config, _, _ = _known_support()
    patches = _owned_scan_patches()
    originals = tuple(patches.items())
    ambient = {name: getattr(config, name) for name in module._HOOKS}
    result = module._scan_outlining_patches(patches)
    if sys.platform not in ("darwin", "linux"):
        assert result is patches
        return
    assert result is not patches
    assert tuple(result) == (*patches, "_post_fusion_custom_pass")
    assert type(result["_post_fusion_custom_pass"]) is module._OutlineScanBodyPass
    assert tuple(patches.items()) == originals
    assert all(result[name] is value for name, value in originals)
    assert {name: getattr(config, name) for name in ambient} == ambient


@pytest.mark.parametrize(
    "kind",
    [
        "chained_donation",
        "foreign_donation",
        "donation_subclass",
        "copy_list",
        "copy_subclass",
        "copy_missing",
        "extra_copy",
        "foreign_post_grad",
        "foreign_post_fusion",
        "foreign_pre_grad",
    ],
)
def test_foreign_scan_hooks_decline_before_qualification(monkeypatch, kind):
    module = _module()
    from dendra.func._scan_donation import _ScanDonationPass
    from dendra.func._scan_inductor import _CopyViewLowering

    patches = _owned_scan_patches()
    donation = patches["_pre_fusion_custom_pass"]
    copy_pass = patches["post_grad_custom_post_pass"][0]

    def foreign(nodes):
        return nodes

    class DonationSubclass(_ScanDonationPass):
        pass

    class CopySubclass(_CopyViewLowering):
        pass

    if kind == "chained_donation":
        donation.previous = foreign
    elif kind == "foreign_donation":
        patches["_pre_fusion_custom_pass"] = foreign
    elif kind == "donation_subclass":
        patches["_pre_fusion_custom_pass"] = DonationSubclass()
    elif kind == "copy_list":
        patches["post_grad_custom_post_pass"] = [copy_pass]
    elif kind == "copy_subclass":
        patches["post_grad_custom_post_pass"] = (CopySubclass(),)
    elif kind == "copy_missing":
        patches["post_grad_custom_post_pass"] = ()
    elif kind == "extra_copy":
        patches["post_grad_custom_post_pass"] = (copy_pass, _CopyViewLowering())
    elif kind == "foreign_post_grad":
        patches["post_grad_custom_post_pass"] = (copy_pass, foreign)
    elif kind == "foreign_post_fusion":
        patches["_post_fusion_custom_pass"] = foreign
    else:
        patches["pre_grad_custom_pass"] = foreign
    originals = tuple(patches.items())

    def unexpected_support():
        raise AssertionError("Foreign scan hooks must retain ordinary codegen")

    monkeypatch.setattr(module, "_support", unexpected_support)
    assert module._scan_outlining_patches(patches) is patches
    assert tuple(patches.items()) == originals


@pytest.mark.parametrize(
    "devices", [("cpu",), ("cpu", "cpu"), ("cpu", "meta"), ("meta",), ()]
)
def test_scan_backend_attaches_outlining_only_for_nonempty_cpu_operands(
    monkeypatch, devices
):
    module = _module()
    from torch._inductor import compile_fx as compile_module

    from dendra.func._scan_inductor import _ScanInductorBackend

    calls = []
    marker = object()

    def attachment(patches):
        calls.append(patches)
        return patches

    monkeypatch.setattr(module, "_scan_outlining_patches", attachment)
    monkeypatch.setattr(compile_module, "compile_fx", lambda *args, **kwargs: marker)
    inputs = [torch.empty(0, device=device) for device in devices]
    inputs.extend((7, None))
    backend = _ScanInductorBackend(None, {"cpp_wrapper": True}, False, "outline_device")
    assert backend(object(), inputs) is marker
    assert bool(calls) is (bool(devices) and all(device == "cpu" for device in devices))


def test_scan_outline_cache_identity_is_portable_stable_and_distinct_from_while():
    module = _module()
    _known_support()
    from torch._inductor.codecache import FxGraphCachePickler

    first = module._OutlineScanBodyPass()
    other = module._OutlineScanBodyPass()
    ordinary = module._OutlineWhileBodyPass()
    assert first.uuid() == other.uuid()
    assert first.uuid() != ordinary.uuid()
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    graph.output(value)
    pickler = FxGraphCachePickler(torch.fx.GraphModule(torch.nn.Module(), graph))
    encoded = pickler.dumps({"_post_fusion_custom_pass": first})
    assert encoded == pickler.dumps({"_post_fusion_custom_pass": other})
    assert encoded != pickler.dumps({"_post_fusion_custom_pass": ordinary})


@pytest.mark.parametrize(
    "nodes,tensors,metadata,expected",
    [(4095, 127, 500, False), (4096, 0, 0, True), (1, 128, 0, True)],
)
def test_body_budget_counts_arithmetic_or_owning_tensor_inputs(
    nodes, tensors, metadata, expected
):
    from types import SimpleNamespace

    from torch._inductor import ir

    _, apply_pass, _ = _known_support()
    loop = _closure(apply_pass)["codegen_while"].cell_contents
    budget = _closure(loop)["large_body"].cell_contents
    inputs = {name: object.__new__(ir.TensorBox) for name in range(tensors)}
    inputs.update({tensors + name: object() for name in range(metadata)})
    graph = SimpleNamespace(
        orig_gm=SimpleNamespace(graph=SimpleNamespace(nodes=range(nodes))),
        graph_inputs=inputs,
    )
    assert budget(graph) is expected


def _run_scan_ad_case():
    """Fresh isolated child: exercise outlined forward and reverse scan bodies."""
    from itertools import combinations

    from torch._dynamo import utils as compiler_utils
    from torch._inductor.codecache import CppWrapperCodeCache

    from dendra.func._scan_compat import install_scan_compatibility
    from dendra.func._scan_inductor import _compile_scan
    from tests.func.test_scan_donation import _inputs, _loss, _program, _reference

    torch.set_num_threads(1)
    module = _module()
    _, apply_pass, _ = _known_support()
    adapter = _closure(apply_pass)["codegen_subgraph"].cell_contents
    enter_body = _closure(adapter)["EnterBody"].cell_contents
    original_budget = module._MIN_BODY_FX_NODES
    # This child has its own compiler cache. Force only the scope-mechanics
    # fixture through the boundary; production retains its normal 4096 budget.
    module._MIN_BODY_FX_NODES = 1
    observed_modes = []
    admissions = []
    original_loader = CppWrapperCodeCache.load_pybinding

    def checked_loader(cls, *arguments, **keywords):
        code = keywords.get("main_code")
        assert type(code) is str
        if "while (1)" in code:
            assert "dendra_while_body_" in code
            assert "__attribute__((noinline))" in code
            admissions.append(True)
        return original_loader(*arguments, **keywords)

    def observed(frame):
        owner = frame.f_locals["self"].wrapper
        observed_modes.append(owner.get_codegened_graph().is_backward)

    install_scan_compatibility()
    compiled = _compile_scan(
        _program("shared", 2),
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {"cpp_wrapper": True, "compile_threads": 1},
        },
    )
    first_activity = None
    prior_output = prior_values = None
    try:
        with (
            torch_compiler_warning_context(),
            patch.object(
                CppWrapperCodeCache, "load_pybinding", classmethod(checked_loader)
            ),
            observe_calls({enter_body.codegen.__code__}, observed),
        ):
            for offset in (0.0, 0.07):
                values = _inputs(4, offset)
                actual = tuple(value.clone().requires_grad_() for value in values)
                expected = tuple(value.clone().requires_grad_() for value in values)
                frozen = _copy_tree(actual)
                versions = tuple(value._version for value in actual)
                output = compiled(*actual)
                reference = _reference(*expected)
                gradients = torch.autograd.grad(_loss(output), actual)
                wanted = torch.autograd.grad(_loss(reference), expected)
                assert output[1].shape == (5, 8)
                for value, correct in zip(output, reference, strict=True):
                    torch.testing.assert_close(value, correct, rtol=2e-11, atol=2e-12)
                for value, correct in zip(gradients, wanted, strict=True):
                    assert (
                        torch.isfinite(value).all() and torch.count_nonzero(correct) > 0
                    )
                    torch.testing.assert_close(value, correct, rtol=2e-10, atol=2e-11)
                torch.testing.assert_close(actual, frozen, rtol=0, atol=0)
                assert tuple(value._version for value in actual) == versions
                for left, right in combinations(output, 2):
                    assert not torch._C._is_alias_of(left, right)
                assert all(
                    not torch._C._is_alias_of(value, operand)
                    for value in output
                    for operand in actual
                )
                if prior_output is not None:
                    torch.testing.assert_close(
                        prior_output, prior_values, rtol=0, atol=0
                    )
                prior_output, prior_values = output, _copy_tree(output)
                activity = (
                    deepcopy(compiler_utils.counters),
                    list(compiler_utils.get_compilation_metrics()),
                    tuple(observed_modes),
                )
                if first_activity is None:
                    assert (
                        admissions
                        and False in observed_modes
                        and True in observed_modes
                    )
                    first_activity = activity
                else:
                    assert activity == first_activity
                protected = [
                    item
                    for item in compiled._dendra_scan_donation_diagnostics()
                    if item.get("guard_applied")
                ]
                assert protected and all(
                    item["is_backward"]
                    and item["is_subgraph"]
                    and item["protected_input_count"] == item["input_count"]
                    and all(value["is_never_reuse"] for value in item["inputs"])
                    for item in protected
                )
        config, _, _ = _known_support()
        assert config._pre_fusion_custom_pass is None
        assert config._post_fusion_custom_pass is None
    finally:
        module._MIN_BODY_FX_NODES = original_budget
    return {
        "qualified": True,
        "forward": True,
        "backward": True,
        "owned_recording": True,
    }


def test_cpp_scan_outline_scope_preserves_reverse_gradients_and_owned_recordings(
    tmp_path,
):
    _known_support()
    if sys.platform not in ("darwin", "linux"):
        pytest.skip("The qualified native CPU wrapper is unavailable")
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCHINDUCTOR_CACHE_DIR=os.fspath(tmp_path / "inductor"),
    )
    script = (
        "import json, runpy, sys; "
        "result = runpy.run_path(sys.argv[1])['_run_scan_ad_case'](); "
        "print('DENDRA_SCAN_OUTLINE_AD=' + json.dumps(result))"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, __file__],
        cwd=Path(dn.__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [
        json.loads(line.removeprefix("DENDRA_SCAN_OUTLINE_AD="))
        for line in completed.stdout.splitlines()
        if line.startswith("DENDRA_SCAN_OUTLINE_AD=")
    ]
    assert rows == [
        {"qualified": True, "forward": True, "backward": True, "owned_recording": True}
    ]


def test_inductor_population_scan_preserves_portable_resume_and_native_wrapper_reuse(
    tmp_path,
):
    _known_support()
    if sys.platform not in ("darwin", "linux"):
        pytest.skip("The qualified native CPU wrapper is unavailable")
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCHINDUCTOR_CACHE_DIR=os.fspath(tmp_path / "inductor"),
    )
    script = (
        "import json, runpy, sys; "
        "result = runpy.run_path(sys.argv[1])['_run_inductor_case'](False, sys.argv[2] == 'reader', 'scan'); "
        "print('DENDRA_SCAN_OUTLINE=' + json.dumps(result))"
    )
    writer_wrappers = None
    for mode in ("writer", "reader"):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                script,
                __file__,
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
            json.loads(line.removeprefix("DENDRA_SCAN_OUTLINE="))
            for line in completed.stdout.splitlines()
            if line.startswith("DENDRA_SCAN_OUTLINE=")
        ]
        assert len(rows) == 1
        row = rows[0]
        assert row["qualified"] and row["network"] is False
        assert row["captures"] == 1 and row["portable"]
        assert row["reader"] is (mode == "reader")
        assert row["native_wrappers"] and row["native_wrapper_loads"]
        assert all(
            row["native_wrappers"][name] == digest
            for name, digest in row["native_wrapper_loads"].items()
        )
        if mode == "reader":
            assert not row["native_wrapper_builds"]
            assert row["native_wrappers"] == writer_wrappers
        else:
            assert row["native_wrapper_builds"]
            writer_wrappers = row["native_wrappers"]


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("plain", True),
        ("boxed_reduction", True),
        ("foreign_graph", False),
        ("no_inputs", False),
        ("scalar_input", False),
        ("two_outputs", False),
        ("input_alias_output", False),
        ("foreign_output", False),
        ("scalar_ir", False),
        ("scatter", False),
        ("foreign_layout", False),
        ("non_bool", False),
        ("non_scalar", False),
    ],
)
def test_condition_scope_requires_a_fresh_owning_scalar_bool(kind, expected):
    from types import SimpleNamespace

    from torch._inductor import ir
    from torch._inductor.graph import GraphLowering

    _, apply_pass, _ = _known_support()
    loop = _closure(apply_pass)["codegen_while"].cell_contents
    predicate = _closure(loop)["_owning_scalar_condition"].cell_contents
    layout = object.__new__(
        ir.FlexibleLayout if kind == "boxed_reduction" else ir.FixedLayout
    )
    layout.dtype, layout._size = torch.bool, []
    operation = object.__new__(ir.ComputedBuffer)
    operation.data = object.__new__(
        ir.Reduction if kind == "boxed_reduction" else ir.Pointwise
    )
    operation.layout = layout
    graph = object.__new__(GraphLowering)
    graph.graph_inputs = {"value": object.__new__(ir.TensorBox)}
    graph.operations, graph.graph_outputs = [operation], [operation]
    if kind == "boxed_reduction":
        output = object.__new__(ir.StorageBox)
        output.data = operation
        graph.graph_outputs = [output]
    elif kind == "foreign_graph":
        graph = SimpleNamespace(**vars(graph))
    elif kind == "no_inputs":
        graph.graph_inputs = {}
    elif kind == "scalar_input":
        graph.graph_inputs["scalar"] = 1
    elif kind == "two_outputs":
        graph.graph_outputs = [operation, operation]
    elif kind == "input_alias_output":
        graph.graph_outputs = [object.__new__(ir.InputBuffer)]
    elif kind == "foreign_output":
        other = object.__new__(ir.ComputedBuffer)
        other.data, other.layout = operation.data, layout
        graph.graph_outputs = [other]
    elif kind == "scalar_ir":
        graph.operations.append(object.__new__(ir.DynamicScalar))
    elif kind == "scatter":
        operation.data = object.__new__(ir.Scatter)
    elif kind == "foreign_layout":
        operation.layout = SimpleNamespace(dtype=torch.bool, _size=[])
    elif kind == "non_bool":
        layout.dtype = torch.float64
    elif kind == "non_scalar":
        layout._size = [1]
    assert predicate(graph) is expected


def _run_condition_false_first_case():
    """Fresh child: a wide predicate escapes its scope before bool extraction."""
    import re

    from torch._dynamo import utils as compiler_utils
    from torch._inductor.codecache import CppWrapperCodeCache

    from dendra.func._while_inductor import _compile_while

    torch.set_num_threads(1)
    _known_support()
    source_checks = []
    loader_code = CppWrapperCodeCache.load_pybinding.__func__.__code__

    def observed(frame):
        code = frame.f_locals["kwargs"].get("main_code")
        assert type(code) is str
        if "while (1)" not in code:
            return
        condition = code.index("// subgraph: while_loop_cond")
        body = code.index("// subgraph: while_loop_body")
        definition = code.rfind("auto dendra_while_body_", 0, condition)
        assert definition >= 0
        name = re.match(r"auto (dendra_while_body_\d+)", code[definition:]).group(1)
        call = code.index(name + "();", condition)
        extraction = code.index("aoti_torch_item_bool", call)
        stop = code.index(" break;", extraction)
        assert definition < condition < call < extraction < stop < body
        predicate = re.search(r"RAIIAtenTensorHandle (\w+_cond_result);", code)
        assert predicate is not None and predicate.start() < code.index("while (1)")
        assert re.search(
            re.escape(predicate.group(1)) + r" = std::move\(", code[condition:call]
        )
        assert code.count("auto dendra_while_body_") >= 2
        source_checks.append(True)

    def program(*values):
        def condition(counter, *state):
            del state
            return counter < 0

        def body(counter, *state):
            return (counter + 1, *(value + 1 for value in state))

        return torch.while_loop(condition, body, values)

    # The real 128-Tensor interface selects the existing generic width budget;
    # this fixture does not lower a production admission threshold.
    inputs = (
        torch.tensor(0, dtype=torch.int64),
        *(torch.tensor(float(index), dtype=torch.float64) for index in range(127)),
    )
    frozen = tuple(value.clone() for value in inputs)
    versions = tuple(value._version for value in inputs)
    rng = torch.get_rng_state().clone()
    compiled = _compile_while(
        program,
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {"cpp_wrapper": True, "compile_threads": 1},
        },
        device=torch.device("cpu"),
    )
    with (
        torch_compiler_warning_context(),
        torch.no_grad(),
        observe_calls({loader_code}, observed),
    ):
        zero = compiled(*inputs)
        assert source_checks
        torch.testing.assert_close(zero, frozen, rtol=0, atol=0)
        counters = deepcopy(compiler_utils.counters)
        metrics = list(compiler_utils.get_compilation_metrics())
        for count in (2, 1, 0):
            values = (torch.tensor(-count, dtype=torch.int64), *inputs[1:])
            actual = compiled(*values)
            torch.testing.assert_close(actual[0], inputs[0], rtol=0, atol=0)
            for value, original in zip(actual[1:], inputs[1:], strict=True):
                torch.testing.assert_close(value, original + count, rtol=0, atol=0)
        assert compiler_utils.counters == counters
        assert list(compiler_utils.get_compilation_metrics()) == metrics
    torch.testing.assert_close(inputs, frozen, rtol=0, atol=0)
    assert tuple(value._version for value in inputs) == versions
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    return {
        "qualified": True,
        "zero_body_skipped": True,
        "two_one_zero_iterations": True,
        "predicate_survives_scope": True,
        "cached_reuse": True,
    }


def test_wide_condition_scope_preserves_false_first_and_cached_nonzero_execution(
    tmp_path,
):
    _known_support()
    if sys.platform not in ("darwin", "linux"):
        pytest.skip("The qualified native CPU wrapper is unavailable")
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCHINDUCTOR_CACHE_DIR=os.fspath(tmp_path / "inductor"),
    )
    script = (
        "import json, runpy, sys; "
        "result = runpy.run_path(sys.argv[1])['_run_condition_false_first_case'](); "
        "print('DENDRA_CONDITION_FALSE_FIRST=' + json.dumps(result))"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, __file__],
        cwd=Path(dn.__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [
        json.loads(line.removeprefix("DENDRA_CONDITION_FALSE_FIRST="))
        for line in completed.stdout.splitlines()
        if line.startswith("DENDRA_CONDITION_FALSE_FIRST=")
    ]
    assert rows == [
        {
            "qualified": True,
            "zero_body_skipped": True,
            "two_one_zero_iterations": True,
            "predicate_survives_scope": True,
            "cached_reuse": True,
        }
    ]
