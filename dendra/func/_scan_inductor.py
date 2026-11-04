"""Local Inductor lowering for copy-view operators in the scan body.

The scan partitioner functionalizes views into copy operators. Lower these
operators after autograd partitioning so the backward scan retains its time
axis even when its derivatives use only loop-invariant operands. Explicit
views followed by independent contiguous clones preserve copy semantics,
conditional operand layouts, and normal Inductor fusion. The graph pass and
authored-derivative decompositions belong only to this backend.
"""

from __future__ import annotations

import torch
from torch._inductor.custom_graph_pass import (
    CustomGraphPass,
    get_custom_graph_passes,
    get_hash_for_files,
)

from dendra._bootstrap import compile_owned, isolate_compilation_callable


class _CopyViewLowering(CustomGraphPass):
    """Expose copy-view layouts after AOT has saved the backward time axis."""

    def uuid(self):
        return get_hash_for_files((__file__,))

    def __call__(self, graph):
        targets = {
            torch.ops.aten.select_copy.int: torch.ops.aten.select.int,
            torch.ops.aten.slice_copy.Tensor: torch.ops.aten.slice.Tensor,
            torch.ops.aten.as_strided_copy.default: torch.ops.aten.as_strided.default,
        }
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in targets:
                continue
            with graph.inserting_before(node):
                view = graph.call_function(targets[node.target], node.args, node.kwargs)
                result = graph.call_function(
                    torch.ops.aten.clone.default,
                    (view,),
                    {"memory_format": torch.contiguous_format},
                )
                # Only the clone has the original copy's layout. Inductor's
                # following fake-tensor update computes the new view metadata.
                result.meta = dict(node.meta)
            node.replace_all_uses_with(result)
            graph.erase_node(node)


class _ScanInductorBackend(torch._TorchCompileInductorWrapper):
    """Keep Inductor's lifecycle with local decompositions and input protection."""

    def __eq__(self, other):
        # Do not let Dynamo equate this lowering with ordinary Inductor merely
        # because their user options match.
        return type(self) is type(other) and super().__eq__(other)

    def __call__(self, graph, inputs, *, config_patches=None):
        from torch._inductor import config
        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.decomposition import select_decomp_table

        patches = {**self.config, **(config_patches or {})}
        with config.patch(patches):
            decompositions = dict(select_decomp_table())
            from ._scan_donation import _scan_donation_pass

            donation_receipts = getattr(self, "_donation_receipts", None)
            if donation_receipts is None:
                donation_receipts = self._donation_receipts = []
            patches["_pre_fusion_custom_pass"] = _scan_donation_pass(
                patches.get("_pre_fusion_custom_pass", config._pre_fusion_custom_pass),
                receipts=donation_receipts,
            )
            previous = patches.get(
                "post_grad_custom_post_pass", config.post_grad_custom_post_pass
            )
            # Keep user callbacks intact, including their inference-aware
            # protocol and cache identity. Torch applies this pass to HOP
            # subgraphs as well as their containing forward/backward graphs.
            patches["post_grad_custom_post_pass"] = (
                *get_custom_graph_passes(previous),
                _CopyViewLowering(),
            )
            # CPU outlining must remain dormant for GPU/mixed-device graphs.
            # Scalar and symbolic operands carry no model-device ownership.
            try:
                tensor_devices = tuple(
                    value.device.type
                    for value in inputs
                    if isinstance(value, torch.Tensor)
                )
            except Exception:
                tensor_devices = ()
            if tensor_devices and all(device == "cpu" for device in tensor_devices):
                from ._while_outlining import _scan_outlining_patches

                patches = _scan_outlining_patches(patches)
        from ._dufort_autograd import _inductor_decompositions

        decompositions.update(_inductor_decompositions())
        return compile_fx(
            graph,
            inputs,
            config_patches=patches,
            decompositions=decompositions,
            compile_region_name=self.name,
        )

    def _donation_diagnostics(self):
        from copy import deepcopy

        return deepcopy(tuple(getattr(self, "_donation_receipts", ())))


def _compile_scan(function, compile_options, *, prefer_cpp_wrapper=False):
    """Use the local lowering only when ordinary Inductor was selected.

    Custom backends, the configured non-Inductor default, disabled compilation,
    and AOTInductor keep their original routes. The incoming option dictionary
    is not modified, including its nested ``options`` dictionary.
    """
    function = isolate_compilation_callable(function)
    options = dict(compile_options)
    if options.get("options") is not None:
        options["options"] = dict(options["options"])
    selected_backend = options.get("backend")
    if selected_backend is None:
        from torch._dynamo.backends.registry import get_default_backend

        selected_backend = get_default_backend()
    inductor_options = dict(options.get("options") or {})
    if (
        selected_backend != "inductor"
        or options.get("disable", False)
        or inductor_options.get("use_aoti", False)
    ):
        return compile_owned(function, **options)
    mode = options.pop("mode", None)
    if mode is not None and options.get("options") is not None:
        raise RuntimeError(
            "Either mode or options can be specified, but both can't be "
            "specified at the same time."
        )
    options.pop("options", None)
    # These options belong to torch.compile, not to Inductor's config parser.
    # Forward them through its outer boundary exactly as for the normal backend.
    outer_options = {}
    for name in ("guard_filter_fn", "use_aoti"):
        if name in inductor_options:
            outer_options[name] = inductor_options.pop(name)
    if outer_options:
        options["options"] = outer_options
    backend = _ScanInductorBackend(
        mode,
        inductor_options,
        options.get("dynamic"),
        options.get("name"),
    )
    if prefer_cpp_wrapper:
        from torch._inductor import config

        # Normalize user options first, including Torch's hyphen aliases. An
        # explicit wrapper choice wins; alternate/disabled C++ code generation
        # also keeps its original route.
        if not (
            backend.config.get("disable_cpp_codegen", config.disable_cpp_codegen)
            or backend.config.get("fx_wrapper", config.fx_wrapper)
        ):
            backend.config.setdefault("cpp_wrapper", True)
    options["backend"] = backend
    compiled = compile_owned(function, **options)
    # Torch's generic callable-backend wrapper forwards reset(), but not the
    # compiler-config provider. Preserve the standard compiled function API.
    compiled.get_compiler_config = backend.get_compiler_config
    compiled._dendra_scan_donation_diagnostics = backend._donation_diagnostics
    return compiled
