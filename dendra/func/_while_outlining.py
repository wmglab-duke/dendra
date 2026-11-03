"""Qualified per-compilation outlining of large CPU inference while bodies.

This private module is inert until an owned CPU compile requests it.
Unknown compiler contracts keep ordinary code generation. No Torch class,
registry, installed source or process-global compiler configuration is changed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import sys
from functools import cache
from pathlib import Path
from types import CodeType, FunctionType, MethodType

from torch._inductor.custom_graph_pass import CustomSchedulerPass

from dendra._tensorify_compat import _code_fingerprint

# Compact bodies benefit from staying inline. Large bodies can make LLVM
# spend minutes optimizing loop-local tensor-handle lifetimes.
_MIN_BODY_FX_NODES = 4096
_STATE = "_dendra_while_body_outline"
_SOURCES = {
    "codegen/cpp_wrapper_cpu.py": "d5225aad881a688d25bcfd5f3ae68e014a0818c4ff7032c079a5910a62a7700c",
    "codegen/wrapper.py": "af08df236c60e8336cc4c55556bae0b2a4bcc5a74e3170ba423cd5451a43894d",
    "graph.py": "999726d1b52d37d398506a90ee28c6b712394beee3636b4c5db3568741faeec8",
    "scheduler.py": "f59778a9d269b256672a05cb819d0c64367076dab89f5fa61735dbccb3662f9d",
    "custom_graph_pass.py": "f0fd5036666de547b6eaafcec274dcf3b17a5d87482218ff4225012f3c754f5b",
    "codegen/cpp.py": "867045db86a698a063a03c7ebbe9b6703f4e07417c3abf552c5f53714f2ccf2e",
    "codegen/common.py": "673589b5af3c0617dadac30608a38dd621fbfa687a2be2ffea440496f59a76e5",
    "ir.py": "710bb5bcbab1b4fb7dff4900bef5a865c84f73abc85babf45579ba25a9d13ff4",
}
_METHODS = {
    "codegen/cpp_wrapper_cpu.py": {
        "CppWrapperCpu": (
            "__init__",
            "create",
            "codegen_while_loop",
            "codegen_subgraph",
            "codegen_subgraph_prefix",
            "codegen_subgraph_suffix",
            "codegen_int_array_var",
            "_codegen_int_array_var_impl",
            "push_kernel_profile_scope_state",
            "pop_kernel_profile_scope_state",
        ),
    },
    "codegen/wrapper.py": {
        "EnterSubgraphLine": ("__post_init__", "codegen"),
        "ExitSubgraphLine": ("__post_init__", "codegen"),
        "EnterKernelProfileScopeLine": ("__post_init__", "codegen"),
        "ExitKernelProfileScopeLine": ("__post_init__", "codegen"),
        "PythonWrapperCodegen": (
            "memory_plan_reuse",
            "_generate",
            "generate_workspace_allocation",
            "writeline",
        ),
    },
    "graph.py": {
        "GraphLowering": ("codegen", "codegen_subgraph", "make_subgraph"),
        "SubgraphLowering": ("__init__", "init_wrapper_code"),
    },
    "codegen/cpp.py": {
        "CppScheduling": ("codegen_node",),
        "CppKernelProxy": ("codegen_nodes",),
        "KernelGroup": ("finalize_kernel", "call_kernel"),
    },
    "ir.py": {
        "FallbackKernel": ("codegen",),
        "NopKernel": ("is_no_op",),
        "ConcatKernel": ("create",),
    },
}
_HOOKS = (
    "_pre_fusion_custom_pass",
    "_post_fusion_custom_pass",
    "pre_grad_custom_pass",
    "post_grad_custom_pre_pass",
    "post_grad_custom_post_pass",
    "joint_custom_pre_pass",
    "joint_custom_post_pass",
)


def _find_code(code, qualname, *, freevars=None):
    pending, matches = [code], []
    while pending:
        current = pending.pop()
        if current.co_qualname == qualname and (
            freevars is None or current.co_freevars == freevars
        ):
            matches.append(current)
        pending.extend(value for value in current.co_consts if type(value) is CodeType)
    if len(matches) != 1:
        raise ValueError("Unknown compiler code structure")
    return matches[0]


def _current(guards):
    for (
        namespace,
        owner_name,
        owner,
        name,
        descriptor,
        function,
        code,
        defaults,
        keyword_defaults,
    ) in guards:
        if (
            namespace.get(owner_name) is not owner
            or vars(owner).get(name) is not descriptor
            or function.__code__ is not code
            or function.__defaults__ is not defaults
        ):
            return False
        actual = function.__kwdefaults__
        if keyword_defaults is None:
            if actual is not None:
                return False
        elif (
            type(actual) is not dict
            or tuple(actual) != tuple(key for key, _ in keyword_defaults)
            or any(actual[key] is not value for key, value in keyword_defaults)
        ):
            return False
    return True


def _config_eligible(config, values, *, active_pass=None):
    def effective(name):
        if name in values:
            return values[name]
        current = config
        for component in name.split("."):
            current = getattr(current, component)
        return current

    backend = effective("cpu_backend")
    return (
        effective("cpp_wrapper") is True
        and effective("cpp_wrapper_build_separate") is False
        and effective("fx_wrapper") is False
        and effective("disable_cpp_codegen") is False
        and effective("memory_planning") is False
        and type(backend) is str
        and backend == "cpp"
        and effective("aot_inductor.allow_stack_allocation") is False
        and effective("aot_inductor.debug_compile") is False
        and all(
            effective(name)
            is (active_pass if name == "_post_fusion_custom_pass" else None)
            for name in _HOOKS
        )
    )


@cache
def _support():
    """Qualify once; later captures cheaply verify loaded function identities."""
    try:
        import torch

        from dendra._bootstrap import torch_compiler_warning_context

        conversion = importlib.import_module("torch._dynamo.convert_frame")
        compile_lock = vars(conversion).get("compile_lock")
        if not hasattr(type(compile_lock), "__enter__") or not hasattr(
            type(compile_lock), "__exit__"
        ):
            return None
        # Match Torch's compiler lock order and serialize the process-global
        # warning-filter scope used only for these lazy compiler imports.
        with compile_lock, torch_compiler_warning_context():
            from torch._inductor import config, ir
            from torch._inductor.codegen import common, cpp, cpp_wrapper_cpu, wrapper
            from torch._inductor.graph import GraphLowering, SubgraphLowering
            from torch._inductor.virtualized import V

        root = Path(torch.__file__).resolve().parent / "_inductor"
        guards = []
        for relative, expected in _SOURCES.items():
            path = root / relative
            source = path.read_bytes()
            if hashlib.sha256(source).hexdigest() != expected:
                return None
            compiled = compile(
                source,
                str(path),
                "exec",
                dont_inherit=True,
                optimize=sys.flags.optimize,
            )
            if relative == "codegen/common.py":
                cpu_factory = _code_fingerprint(
                    _find_code(
                        compiled,
                        "_init_builtin_backend_registration.<locals>.<lambda>",
                        freevars=("cpu_backends",),
                    )
                )
            module = importlib.import_module(
                "torch._inductor." + relative[:-3].replace("/", ".")
            )
            namespace = vars(module)
            for owner_name, names in _METHODS.get(relative, {}).items():
                owner = namespace[owner_name]
                for name in names:
                    descriptor = vars(owner)[name]
                    function = (
                        descriptor.__func__
                        if type(descriptor) in (staticmethod, classmethod)
                        else descriptor
                    )
                    if (
                        type(function) is not FunctionType
                        or function.__globals__ is not namespace
                    ):
                        return None
                    code = function.__code__
                    if _code_fingerprint(code) != _code_fingerprint(
                        _find_code(compiled, owner_name + "." + name)
                    ):
                        return None
                    guards.append(
                        (
                            namespace,
                            owner_name,
                            owner,
                            name,
                            descriptor,
                            function,
                            code,
                            function.__defaults__,
                            None
                            if function.__kwdefaults__ is None
                            else tuple(function.__kwdefaults__.items()),
                        )
                    )
        cpu_type = cpp_wrapper_cpu.CppWrapperCpu
        computed_type, fallback_type = ir.ComputedBuffer, ir.FallbackKernel
        concat_type, nop_is_noop = ir.ConcatKernel, ir.NopKernel.is_no_op
        loop_types = (ir.Pointwise, ir.Reduction, ir.Scatter)
        cat = torch.ops.aten.cat.default
        original_writer = wrapper.PythonWrapperCodegen.writeline
        device_codegen_type = common.DeviceCodegen
    except Exception:
        return None

    def stock_backend():
        if type(common.device_codegens) is not dict:
            return False
        registration = common.device_codegens.get("cpu")
        if type(registration) is not device_codegen_type:
            return False
        factory = registration.scheduling
        if (
            type(factory) is not FunctionType
            or factory.__globals__ is not vars(common)
            or _code_fingerprint(factory.__code__) != cpu_factory
            or factory.__closure__ is None
            or len(factory.__closure__) != 1
        ):
            return False
        backends = factory.__closure__[0].cell_contents
        return (
            type(backends) is dict
            and backends.get("cpp") is cpp.CppScheduling
            and registration.cpp_wrapper_codegen is cpu_type
        )

    def workspace_free(graph):
        if type(graph) not in (GraphLowering, SubgraphLowering):
            return False
        for operation in graph.operations:
            if type(operation) is computed_type:
                if type(operation.data) not in loop_types:
                    return False
            elif type(operation) is fallback_type:
                if operation.op_overload is not cat:
                    return False
            elif type(operation) is concat_type:
                if type(operation).is_no_op is not nop_is_noop:
                    return False
            else:
                return False
        return True

    @dataclasses.dataclass
    class EnterBody(wrapper.EnterKernelProfileScopeLine):
        name: str

        def codegen(self, code):
            self.wrapper.push_kernel_profile_scope_state()
            code.writeline(f"auto {self.name} = [&]() __attribute__((noinline)) {{")
            code.do_indent()

    @dataclasses.dataclass
    class ExitBody(wrapper.ExitKernelProfileScopeLine):
        name: str

        def codegen(self, code):
            self.wrapper.pop_kernel_profile_scope_state()
            code.do_unindent()
            code.writeline("};")
            code.writeline(f"{self.name}();")

    def codegen_while(owner, while_loop, stack_output=False):
        state = vars(owner)[_STATE]
        try:
            body = while_loop.body_subgraph.graph
            eligible = (
                not stack_output
                and _current(guards)
                and stock_backend()
                and not owner.allocated_workspaces
                and len(body.orig_gm.graph.nodes) >= _MIN_BODY_FX_NODES
                and workspace_free(body)
                and workspace_free(while_loop.cond_subgraph.graph)
                and owner.writeline.__func__ is original_writer
            )
        except Exception:
            eligible = False
        if not eligible:
            return state["while"](while_loop, stack_output=stack_output)
        state["bodies"].append(while_loop.body_subgraph)
        try:
            return state["while"](while_loop, stack_output=stack_output)
        finally:
            state["bodies"].pop()

    def codegen_subgraph(owner, subgraph, outer_inputs, outer_outputs):
        state = vars(owner)[_STATE]
        if not state["bodies"] or subgraph is not state["bodies"][-1]:
            return state["subgraph"](subgraph, outer_inputs, outer_outputs)
        name = "dendra_while_body_" + str(state["next"])
        state["next"] += 1
        # The admitted stock C++ loops / aten.cat paths cannot allocate a
        # persistent wrapper workspace. Unknown IR declines before this point.
        owner.push_kernel_profile_scope_state()
        try:
            owner.writeline(EnterBody(owner, name))
            try:
                return state["subgraph"](subgraph, outer_inputs, outer_outputs)
            finally:
                owner.writeline(ExitBody(owner, name))
        finally:
            owner.pop_kernel_profile_scope_state()

    def apply(custom_pass, nodes):
        graph, owner = V.graph, V.graph.wrapper_code
        try:
            eligible = (
                _current(guards)
                and stock_backend()
                and type(owner) is cpu_type
                and owner.device == "cpu"
                and graph.device_type == "cpu"
                and graph.is_inference
                and not graph.aot_mode
                and not graph.is_dual_wrapper_mode
                and _config_eligible(config, {}, active_pass=custom_pass)
                and cpp.CppScheduling.kernel_proxy_cls is cpp.CppKernelProxy
                and cpp.CppKernelProxy.kernel_cls is cpp.CppKernel
                and cpp.CppKernelProxy.vec_kernel_cls is cpp.CppVecKernel
                and cpp.CppKernelProxy.tile2d_kernel_cls is cpp.CppTile2DKernel
            )
        except Exception:
            eligible = False
        if not eligible or _STATE in vars(owner):
            return nodes
        if "codegen_while_loop" in vars(owner) or "codegen_subgraph" in vars(owner):
            return nodes
        setattr(
            owner,
            _STATE,
            {
                "while": owner.codegen_while_loop,
                "subgraph": owner.codegen_subgraph,
                "bodies": [],
                "next": 0,
            },
        )
        owner.codegen_while_loop = MethodType(codegen_while, owner)
        owner.codegen_subgraph = MethodType(codegen_subgraph, owner)
        return nodes

    return config, apply, guards


class _OutlineWhileBodyPass(CustomSchedulerPass):
    """Stateless, import-addressable pass so portable caches can pickle it."""

    def uuid(self):
        support = _support()
        if support is None or not _current(support[2]):
            return None
        from torch._inductor.custom_graph_pass import get_hash_for_files

        root = Path(sys.modules["torch"].__file__).resolve().parent / "_inductor"
        return get_hash_for_files(
            (
                str(Path(__file__).resolve()),
                *(str(root / name) for name in sorted(_SOURCES)),
            ),
            extra="dendra-cpu-while-body-4096-v1",
        )

    def __call__(self, nodes):
        support = _support()
        if support is None or not _current(support[2]):
            return nodes
        return support[1](self, nodes)


def _while_outlining_options(options, *, device):
    """Return copied compile options with local outlining only when qualified."""
    if (
        sys.platform not in ("darwin", "linux")
        or getattr(device, "type", None) != "cpu"
        or type(options.get("backend")) is not str
        or options.get("backend") != "inductor"
        or options.get("disable", False) is not False
        or options.get("mode") is not None
    ):
        return options
    requested = options.get("options") or {}
    if any(type(name) is not str for name in requested):
        return options
    values = {name.replace("-", "_"): value for name, value in requested.items()}
    if "use_aoti" in values or any(name in values for name in _HOOKS):
        return options
    try:
        from torch._inductor import config

        if not _config_eligible(config, values):
            return options
    except Exception:
        return options
    support = _support()
    if support is None:
        return options
    config, _apply, guards = support
    try:
        if not _current(guards) or not _config_eligible(config, values):
            return options
    except Exception:
        return options
    result = dict(options)
    result["options"] = {
        **requested,
        "_post_fusion_custom_pass": _OutlineWhileBodyPass(),
    }
    return result
