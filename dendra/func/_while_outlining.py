"""Qualified per-compilation outlining of large CPU while and scan bodies.

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
# AD scans carry saved intermediates through a wide owning-handle interface,
# which can be expensive even with a compact arithmetic graph.
_MIN_BODY_TENSOR_INPUTS = 128
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
            "codegen_tensor_item",
            "_codegen_dynamic_scalar",
        ),
    },
    "codegen/wrapper.py": {
        "EnterSubgraphLine": ("__post_init__", "codegen"),
        "ExitSubgraphLine": ("__post_init__", "codegen"),
        "EnterKernelProfileScopeLine": ("__post_init__", "codegen"),
        "ExitKernelProfileScopeLine": ("__post_init__", "codegen"),
        "DynamicScalarLine": ("codegen",),
        "UnbackedSymbolDefsLine": ("codegen",),
        "PythonWrapperCodegen": (
            "memory_plan_reuse",
            "_generate",
            "generate_workspace_allocation",
            "writeline",
            "codegen_dynamic_scalar",
            "codegen_unbacked_symbol_decl",
            "_codegen_unbacked_symbol_defs_for_outputs",
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
        "DynamicScalar": ("__init__", "should_allocate", "codegen"),
        "AssertScalar": ("__init__", "should_allocate", "has_side_effects", "codegen"),
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


def _config_eligible(config, values, *, active_pass=None, scan_owned=False):
    def effective(name):
        if name in values:
            return values[name]
        current = config
        for component in name.split("."):
            current = getattr(current, component)
        return current

    hooks = {"_post_fusion_custom_pass": active_pass}
    if scan_owned:
        # These two passes are part of scan's ownership and AD contract. They
        # must remain intact; arbitrary caller callbacks retain stock codegen.
        from ._scan_donation import _ScanDonationPass
        from ._scan_inductor import _CopyViewLowering

        donation = effective("_pre_fusion_custom_pass")
        post_grad = effective("post_grad_custom_post_pass")
        if (
            type(donation) is not _ScanDonationPass
            or donation.previous is not None
            or type(post_grad) is not tuple
            or len(post_grad) != 1
            or type(post_grad[0]) is not _CopyViewLowering
        ):
            return False
        hooks["_pre_fusion_custom_pass"] = donation
        hooks["post_grad_custom_post_pass"] = post_grad

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
        and all(effective(name) is hooks.get(name) for name in _HOOKS)
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
        scalar_types = (ir.DynamicScalar, ir.AssertScalar)
        tensor_box_type = ir.TensorBox
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
            elif type(operation) in scalar_types:
                # Scan's loop index extraction and bounds assertions emit
                # scalar declarations / checks without persistent workspaces.
                pass
            else:
                return False
        return True

    def large_body(graph):
        return (
            len(graph.orig_gm.graph.nodes) >= _MIN_BODY_FX_NODES
            or sum(
                isinstance(value, tensor_box_type)
                for value in graph.graph_inputs.values()
            )
            >= _MIN_BODY_TENSOR_INPUTS
        )

    def push_scope(owner):
        owner.push_kernel_profile_scope_state()
        state = vars(owner)[_STATE]
        state.setdefault("scalar_decls", []).append(owner.unbacked_symbol_decls)
        owner.unbacked_symbol_decls = owner.unbacked_symbol_decls.copy()

    def pop_scope(owner):
        owner.unbacked_symbol_decls = vars(owner)[_STATE]["scalar_decls"].pop()
        owner.pop_kernel_profile_scope_state()

    @dataclasses.dataclass
    class EnterBody(wrapper.EnterKernelProfileScopeLine):
        name: str

        def codegen(self, code):
            push_scope(self.wrapper)
            code.writeline(f"auto {self.name} = [&]() __attribute__((noinline)) {{")
            code.do_indent()

    @dataclasses.dataclass
    class ExitBody(wrapper.ExitKernelProfileScopeLine):
        name: str

        def codegen(self, code):
            pop_scope(self.wrapper)
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
                and large_body(body)
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
        # Admitted loops, scalar checks and aten.cat cannot allocate persistent
        # wrapper workspaces. Unknown IR declines before this point.
        push_scope(owner)
        try:
            owner.writeline(EnterBody(owner, name))
            try:
                return state["subgraph"](subgraph, outer_inputs, outer_outputs)
            finally:
                owner.writeline(ExitBody(owner, name))
        finally:
            pop_scope(owner)

    def apply(custom_pass, nodes):
        graph, owner = V.graph, V.graph.wrapper_code
        scan_owned = type(custom_pass) is _OutlineScanBodyPass
        try:
            eligible = (
                _current(guards)
                and stock_backend()
                and type(owner) is cpu_type
                and owner.device == "cpu"
                and graph.device_type == "cpu"
                and (
                    graph.is_inference
                    or (
                        scan_owned
                        and type(graph.is_inference) is bool
                        and type(graph.is_backward) is bool
                    )
                )
                and not graph.aot_mode
                and not graph.is_dual_wrapper_mode
                and _config_eligible(
                    config, {}, active_pass=custom_pass, scan_owned=scan_owned
                )
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
        scan_owned = type(self) is _OutlineScanBodyPass
        local_sources = (Path(__file__).resolve(),)
        if scan_owned:
            local_sources += tuple(
                Path(__file__).with_name(name).resolve()
                for name in (
                    "_scan_inductor.py",
                    "_scan_donation.py",
                    "_scan_compat.py",
                )
            )
        return get_hash_for_files(
            (
                *(str(path) for path in local_sources),
                *(str(root / name) for name in sorted(_SOURCES)),
            ),
            extra=(
                "dendra-cpu-scan-body-4096fx-or-128inputs-v2"
                if scan_owned
                else "dendra-cpu-while-body-4096fx-or-128inputs-v2"
            ),
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


class _OutlineScanBodyPass(_OutlineWhileBodyPass):
    """Reuse the body boundary for scan-owned inference and first-order AD."""


def _scan_outlining_patches(patches):
    """Add local outlining only for the exact scan-owned compiler pass bundle."""
    if sys.platform not in ("darwin", "linux"):
        return patches
    try:
        from torch._inductor import config

        if not _config_eligible(config, patches, scan_owned=True):
            return patches
    except Exception:
        return patches
    support = _support()
    if support is None:
        return patches
    config, _apply, guards = support
    try:
        if not _current(guards) or not _config_eligible(
            config, patches, scan_owned=True
        ):
            return patches
    except Exception:
        return patches
    return {**patches, "_post_fusion_custom_pass": _OutlineScanBodyPass()}
