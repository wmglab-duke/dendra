"""Local Inductor lowering for copy-view operators in the scan body.

The scan partitioner functionalizes views into copy operators. Inductor does
not decompose select_copy/slice_copy in the verified Torch releases, leaving
opaque copy/dispatch boundaries between otherwise fusible numerical kernels.
An owned decomposition table preserves their copy semantics while allowing
normal Inductor fusion. Explicit storage views also retain conditional operand
layout constraints after scan functionalization. Nothing is registered or
patched globally.
"""

from __future__ import annotations

import torch

from dendra._bootstrap import isolate_compilation_callable


def _select_copy(value, dim, index):
    return torch.ops.aten.select.int(value, dim, index).clone(
        memory_format=torch.contiguous_format
    )


def _slice_copy(value, dim=0, start=None, end=None, step=1):
    return torch.ops.aten.slice.Tensor(value, dim, start, end, step).clone(
        memory_format=torch.contiguous_format
    )


def _as_strided_copy(value, size, stride, storage_offset=None):
    # The FX-visible storage view constrains its producer's exact strides.
    # Inductor's direct copy-view lowering otherwise hides that requirement
    # from outer conditional graphs after scan's view functionalization.
    return torch.ops.aten.as_strided.default(value, size, stride, storage_offset).clone(
        memory_format=torch.contiguous_format
    )


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
        decompositions[torch.ops.aten.select_copy.int] = _select_copy
        decompositions[torch.ops.aten.slice_copy.Tensor] = _slice_copy
        decompositions[torch.ops.aten.as_strided_copy.default] = _as_strided_copy
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
        return torch.compile(function, **options)
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
    compiled = torch.compile(function, **options)
    # Torch's generic callable-backend wrapper forwards reset(), but not the
    # compiler-config provider. Preserve the standard compiled function API.
    compiled.get_compiler_config = backend.get_compiler_config
    compiled._dendra_scan_donation_diagnostics = backend._donation_diagnostics
    return compiled
