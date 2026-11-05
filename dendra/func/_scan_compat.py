"""Explicit compatibility activation for Dendra's experimental scan executor.

PyTorch 2.14's HOP partitioner supplies one cotangent slot for every output,
including integer and Boolean outputs. Its shared backward helper otherwise
filters those outputs, producing mismatched positional signatures. The repair
below gives only the partitioner a private backward helper with the full mask.
Shared utility functions and their existing callers are never replaced.

Joint capture also keeps fake backward on one autograd thread. Mixed CPU/CUDA
operands otherwise let engine workers share a FakeTensorMode whose kernel flag
is not thread-local, racing with the dispatch state on those worker threads.
Disposable capture saves are isolated from caller activation-checkpoint hooks;
those hooks must observe real execution, not replay while tracing fake backward.

Discrete operands retain their values during joint materialization. Torch's
shared helper substitutes uninitialized storage for integer inputs; without a
FakeTensorMode those examples become invalid callback indices/frame counters.
All examples retain their authored view strides, offsets, and lazy flags so
explicit strided views use the same storage frame during backward capture.
Real eager operands also receive a scoped FakeTensorMode during disposable
materialization, so copied examples cannot execute native resource-owning ops.

Activation is persistent and process-wide at that partitioner entry point:
backward and checkpoint replay can run after the selecting forward returns.
Importing this module performs no activation and imports no private Torch HOP.
"""

from __future__ import annotations

import dis
import importlib
import inspect
import threading
from functools import update_wrapper
from types import CodeType, FunctionType, SimpleNamespace

import torch

from dendra._bootstrap import torch_compiler_warning_context

from ._types import FunctionalizationError

_INSTALL_LOCK = threading.Lock()
_INSTALLED = None
_INDUCTOR_LOCK = threading.Lock()
_INDUCTOR_CONTRACT = None


def _unsupported(details):
    return FunctionalizationError(
        "Dendra scan execution cannot establish the required PyTorch compiler "
        f"correctness contract: {details}. The selected scan is not replaced "
        "with an unrolled execution."
    )


def _owned_callable(module, owner, name, *, args=None, kwargs=None):
    """Check an owned Python call seam, not a release or source fingerprint."""
    function = getattr(owner, name, None)
    if (
        not isinstance(function, FunctionType)
        or function.__globals__ is not module.__dict__
        or function.__module__ != module.__name__
    ):
        raise _unsupported(f"{owner.__name__}.{name} has already been replaced")
    if args is not None:
        try:
            inspect.signature(function).bind(*args, **(kwargs or {}))
        except (TypeError, ValueError) as error:
            raise _unsupported(
                f"{owner.__name__}.{name} has an incompatible signature"
            ) from error
    return function


def _loads_global(function, name):
    pending = [function.__code__]
    while pending:
        code = pending.pop()
        if any(
            instruction.opname == "LOAD_GLOBAL" and instruction.argval == name
            for instruction in dis.get_instructions(code)
        ):
            return True
        pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
    return False


def _check_references(references):
    for owner, name, original in references:
        if getattr(owner, name, None) is not original:
            raise _unsupported(f"the active helper {owner.__name__}.{name} changed")


def _validate_scan_contract():
    names = (
        "torch._higher_order_ops.utils",
        "torch._higher_order_ops.partitioner",
        "torch._functorch._aot_autograd.graph_capture_wrappers",
        "torch._higher_order_ops.scan",
        "torch._guards",
        "torch._subclasses.fake_tensor",
    )
    try:
        modules = {name: importlib.import_module(name) for name in names}
        aot = importlib.import_module("torch._functorch.aot_autograd")
    except ImportError as error:
        raise _unsupported(
            f"required scan/AOT interface is unavailable: {error}"
        ) from error

    utils = modules["torch._higher_order_ops.utils"]
    partitioner = modules["torch._higher_order_ops.partitioner"]
    wrappers = modules["torch._functorch._aot_autograd.graph_capture_wrappers"]
    scan = modules["torch._higher_order_ops.scan"]
    placeholder = object()
    functions = (
        (utils, "create_bw_fn", (placeholder, ()), {"return_fw_outputs": True}),
        (utils, "prepare_fw_with_masks_all_requires_grad", (placeholder,), {}),
        (utils, "_clone_aliasing_output", ((), ()), {}),
        (utils, "_from_fun", (placeholder,), {}),
        (utils, "materialize_as_graph", (placeholder, ()), {"force_enable_grad": True}),
        (
            partitioner,
            "create_hop_joint_graph",
            (placeholder, ()),
            {"functionalize": True},
        ),
        (wrappers, "create_joint", (placeholder,), {"aot_config": placeholder}),
        (scan, "scan_autograd", (placeholder, (), (), ()), {}),
    )
    references = []
    for module, name, args, kwargs in functions:
        function = _owned_callable(module, module, name, args=args, kwargs=kwargs)
        references.append((module, name, function))
    partitioner_type = getattr(partitioner, "HopGraphMinCutPartitioner", None)
    if (
        not isinstance(partitioner_type, type)
        or partitioner_type.__module__ != partitioner.__name__
        or getattr(scan, "HopGraphMinCutPartitioner", None) is not partitioner_type
    ):
        raise _unsupported("scan no longer uses the expected joint-graph partitioner")
    partition = _owned_callable(
        partitioner,
        partitioner_type,
        "create_partitioned_graph",
        args=(placeholder, ()),
    )
    if not _loads_global(partition, "create_hop_joint_graph"):
        raise _unsupported("the partitioner no longer uses the joint-graph entry point")
    references.extend(
        (
            (partitioner, "HopGraphMinCutPartitioner", partitioner_type),
            (scan, "HopGraphMinCutPartitioner", partitioner_type),
            (partitioner_type, "create_partitioned_graph", partition),
        )
    )
    for function, globals_needed in (
        (partitioner.create_hop_joint_graph, ("create_bw_fn", "materialize_as_graph")),
        (utils.materialize_as_graph, ("_from_fun",)),
    ):
        if not all(_loads_global(function, name) for name in globals_needed):
            raise _unsupported(
                f"{function.__name__} no longer exposes the required helper seams"
            )
    if partitioner.create_bw_fn is not utils.create_bw_fn:
        raise _unsupported("the partitioner's backward helper has already changed")
    references.append((partitioner, "create_bw_fn", partitioner.create_bw_fn))
    if partitioner.materialize_as_graph is not utils.materialize_as_graph:
        raise _unsupported("the partitioner's graph materializer has already changed")
    references.append(
        (partitioner, "materialize_as_graph", partitioner.materialize_as_graph)
    )
    if aot.create_joint is not wrappers.create_joint:
        raise _unsupported("the AOT joint entry point has already changed")
    references.append((aot, "create_joint", aot.create_joint))
    if not isinstance(getattr(utils, "FunctionalTensor", None), type):
        raise _unsupported("functional-tensor unwrapping is unavailable")
    references.append((utils, "FunctionalTensor", utils.FunctionalTensor))
    config = getattr(aot, "AOTConfig", None)
    try:
        inspect.signature(config).bind(
            fw_compiler=None,
            bw_compiler=None,
            partition_fn=None,
            decompositions={},
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=False,
        )
    except (TypeError, ValueError) as error:
        raise _unsupported(
            "AOTConfig cannot express the required joint capture"
        ) from error
    references.append((aot, "AOTConfig", config))
    guards = modules["torch._guards"]
    detect = _owned_callable(guards, guards, "detect_fake_mode", args=((),))
    references.append((guards, "detect_fake_mode", detect))
    fake_module = modules["torch._subclasses.fake_tensor"]
    fake_type = getattr(fake_module, "FakeTensorMode", None)
    if not isinstance(fake_type, type) or fake_type.__module__ != fake_module.__name__:
        raise _unsupported("the metadata-only materialization mode changed")
    references.append((fake_module, "FakeTensorMode", fake_type))
    for name, args, kwargs in (
        ("__init__", (placeholder,), {"allow_non_fake_inputs": True}),
        ("__enter__", (placeholder,), {}),
        ("__exit__", (placeholder, None, None, None), {}),
    ):
        function = _owned_callable(
            fake_module, fake_type, name, args=args, kwargs=kwargs
        )
        references.append((fake_type, name, function))
    return modules, tuple(references), aot.AOTConfig, aot.create_joint


def _probe_scan_contract(entry):
    """Qualify the private adapter once, before any process-wide assignment.

    This tiny CPU joint capture tests interleaved cotangent slots, integer
    index values, Boolean values, gradients and storage independence. It does
    not compile a simulation or select a release-specific implementation.
    """
    try:
        with (
            torch.utils._python_dispatch._disable_current_modes(),
            torch._C._DisableFuncTorch(),
            torch.inference_mode(False),
            torch.enable_grad(),
            torch.autocast("cpu", enabled=False),
            torch.autograd.set_multithreading_enabled(False),
            torch_compiler_warning_context(),
        ):
            x = torch.tensor(
                [0.2, -0.4], dtype=torch.float64, device="cpu", requires_grad=True
            )
            y = torch.tensor(
                [0.7, 0.9], dtype=torch.float64, device="cpu", requires_grad=True
            )
            index = torch.tensor([1, 0], dtype=torch.int64, device="cpu")
            flag = torch.tensor([True, False], device="cpu")
            primals = (x, y, index, flag)

            def forward(x, y, index, flag):
                return index + 1, x.square() + y.index_select(0, index), ~flag, x * y

            outputs = forward(*primals)
            tangents = (
                torch.zeros_like(index),
                torch.tensor([1.3, -0.6], dtype=x.dtype, device="cpu"),
                torch.zeros_like(flag),
                torch.tensor([-0.4, 0.8], dtype=x.dtype, device="cpu"),
            )
            joint = entry(forward, primals, functionalize=True)
            if joint.n_primals != 4 or joint.n_fw_outputs != 4 or joint.n_tangents != 4:
                raise AssertionError("The joint graph lost positional tensor slots")
            result = tuple(joint.joint_gm(*primals, *tangents))
            gradients = (
                2 * x * tangents[1] + y * tangents[3],
                tangents[1].flip(0) + x * tangents[3],
                torch.zeros_like(index),
                torch.zeros_like(flag),
            )
            torch.testing.assert_close(
                result, (*outputs, *gradients), rtol=1e-12, atol=1e-12
            )
            for offset, value in enumerate(result):
                others = (*primals, *tangents, *result[:offset])
                if any(torch._C._is_alias_of(value, other) for other in others):
                    raise AssertionError(
                        "The joint graph returned aliased output storage"
                    )
    except Exception as error:
        raise _unsupported(
            f"the mixed-output joint behavior probe failed: {error}"
        ) from error


def validate_inductor_scan_contract():
    """Validate input-ownership protection, independent of a Torch patch label.

    The reusable-buffer protocol is qualified once. Subsequent calls check
    saved callable/class identities only. Actual graph fields are checked by
    the guard before it mutates that graph's input-protection set.
    """
    global _INDUCTOR_CONTRACT

    with _INDUCTOR_LOCK:
        if _INDUCTOR_CONTRACT is not None:
            _check_references(_INDUCTOR_CONTRACT)
            return
        try:
            graph = importlib.import_module("torch._inductor.graph")
            scheduler = importlib.import_module("torch._inductor.scheduler")
            wrapper = importlib.import_module("torch._inductor.codegen.wrapper")
            virtualized = importlib.import_module("torch._inductor.virtualized")
            owners = (
                (graph, "GraphLowering"),
                (graph, "SubgraphLowering"),
                (scheduler, "BaseSchedulerNode"),
                (wrapper, "PythonWrapperCodegen"),
            )
            references = []
            for module, name in owners:
                owner = getattr(module, name, None)
                if not isinstance(owner, type) or owner.__module__ != module.__name__:
                    raise _unsupported(f"{module.__name__}.{name} has changed")
                references.append((module, name, owner))
            placeholder = object()
            methods = (
                (graph, graph.GraphLowering, "__init__", (placeholder, placeholder)),
                (graph, graph.GraphLowering, "make_subgraph", (placeholder,) * 4),
                (graph, graph.GraphLowering, "placeholder", (placeholder,) * 4),
                (graph, graph.SubgraphLowering, "__init__", (placeholder,) * 2),
                (
                    wrapper,
                    wrapper.PythonWrapperCodegen,
                    "can_reuse",
                    (placeholder,) * 2,
                ),
                (
                    scheduler,
                    scheduler.BaseSchedulerNode,
                    "decide_inplace_update",
                    (placeholder,),
                ),
            )
            for module, owner, name, args in methods:
                function = _owned_callable(module, owner, name, args=args)
                references.append((owner, name, function))
            can_reuse = wrapper.PythonWrapperCodegen.can_reuse
            if "can_free" in can_reuse.__code__.co_names:
                can_free = _owned_callable(
                    wrapper,
                    wrapper.PythonWrapperCodegen,
                    "can_free",
                    args=(placeholder, placeholder),
                )
                references.append((wrapper.PythonWrapperCodegen, "can_free", can_free))
            if (
                "can_reuse"
                not in scheduler.BaseSchedulerNode.decide_inplace_update.__code__.co_names
            ):
                raise _unsupported(
                    "in-place planning no longer consults the reusable-buffer protocol"
                )
            test_graph = SimpleNamespace(
                removed_buffers=set(),
                graph_inputs={},
                constants={},
                torchbind_constants={},
                never_reuse_buffers=set(),
                never_reuse_but_free_buffers=set(),
            )
            test_wrapper = object.__new__(wrapper.PythonWrapperCodegen)
            test_wrapper.freed = set()
            buffer = SimpleNamespace(get_name=lambda: "dendra_contract_input")
            with virtualized.V.set_graph_handler(test_graph):
                if not test_wrapper.can_reuse(buffer):
                    raise _unsupported(
                        "the reusable-buffer behavior probe is not applicable"
                    )
                test_graph.never_reuse_buffers.add(buffer.get_name())
                if test_wrapper.can_reuse(buffer):
                    raise _unsupported("input protection does not prevent buffer reuse")
        except FunctionalizationError:
            raise
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            raise _unsupported(
                f"the input-ownership interface is incompatible: {error}"
            ) from error
        _INDUCTOR_CONTRACT = tuple(references)


def _make_value_preserving_materializer(utils):
    """Isolate scan examples, retaining view metadata and fake op dispatch."""
    # Functional-tensor unwrapping below is adapted from PyTorch's _from_fun.
    # See THIRD_PARTY_NOTICES.md and the upstream license in LICENSES/PyTorch.txt.
    from ._hop_inputs import _copy_view

    original_from_fun = utils._from_fun

    def from_fun_preserving_metadata(value):
        if not isinstance(value, torch.Tensor):
            return original_from_fun(value)
        if value.dtype.is_floating_point or value.dtype.is_complex:
            example = original_from_fun(value)
            offset = value.storage_offset()
            conjugate, negative = value.is_conj(), value.is_neg()
            if offset == 0 and not conjugate and not negative:
                return example
            # Torch's empty_strided examples preserve strides but discard the
            # original offset and lazy flags. Explicit authored as_strided ops
            # then differentiate against a different input storage frame.
            # These are metadata-only examples under the materializer's fake
            # mode; no source values or native resources are evaluated.
            size = (
                offset
                + 1
                + sum(
                    (length - 1) * stride
                    for length, stride in zip(value.shape, value.stride(), strict=True)
                )
                if value.numel()
                else 0
            )
            example = example.new_empty((size,), requires_grad=False).as_strided(
                value.shape, value.stride(), offset
            )
            if negative:
                example = torch._neg_view(example)
            if conjugate:
                example = example.conj()
            return example.requires_grad_(value.requires_grad)
        # Unwrap discrete operands before copying their addressed storage
        # span. Floating losses can also read discrete strided views, so their
        # numerical values and original layout both belong to the example.
        if isinstance(value, utils.FunctionalTensor):
            from torch._functorch.aot_autograd import from_fun

            torch._sync(value)
            value = from_fun(value)
        elif torch._is_functional_tensor(value):
            torch._sync(value)
            value = torch._from_functional_tensor(value)
        return _copy_view(value)

    original = utils.materialize_as_graph
    private_globals = dict(original.__globals__)
    private_globals["_from_fun"] = from_fun_preserving_metadata
    materialize = FunctionType(
        original.__code__,
        private_globals,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    update_wrapper(materialize, original)
    materialize.__kwdefaults__ = original.__kwdefaults__

    def materialize_with_fake_examples(*args, **kwargs):
        from torch._guards import detect_fake_mode
        from torch._subclasses.fake_tensor import FakeTensorMode

        # Real eager scan operands need the same metadata-only materialization
        # as Dynamo's fake operands. Fabricated/copied examples must not execute
        # native kernels with identity-owned opaque resources (for example a
        # solver topology token). Reuse the caller's mode when it exists; this
        # context is confined to disposable graph capture, not real execution.
        fake_mode = detect_fake_mode((args, kwargs))
        if fake_mode is None:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        with fake_mode:
            return materialize(*args, **kwargs)

    # Keep all helper dependencies inside the same private globals inventory.
    # No shared Torch helper or user tensor is replaced by this adapter.
    guarded = FunctionType(
        materialize_with_fake_examples.__code__,
        private_globals,
        original.__name__,
        materialize_with_fake_examples.__defaults__,
        materialize_with_fake_examples.__closure__,
    )
    update_wrapper(guarded, original)
    return guarded


def _capture_saved_tensor_identity(tensor):
    return tensor


def _make_full_output_backward(utils, aot_config_type, create_joint):
    # Adapted from PyTorch's torch/_higher_order_ops/utils.py:create_bw_fn.
    # See THIRD_PARTY_NOTICES.md and the upstream license in LICENSES/PyTorch.txt.
    prepare_original = utils.prepare_fw_with_masks_all_requires_grad
    clone_aliases = utils._clone_aliasing_output

    def prepare_full_output_masks(function):
        prepared = prepare_original(function)

        def forward_with_masks(*args):
            outputs, _differentiable_mask = prepared(*args)
            # The partitioner has already verified a flat Tensor output list.
            # Preserve every positional slot; AOT's subsequent requires_grad
            # check still excludes integer/Boolean leaves from autograd.grad.
            return outputs, [True] * len(outputs)

        return forward_with_masks

    def create_full_output_backward(function, args, return_fw_outputs=False):
        config = aot_config_type(
            fw_compiler=None,
            bw_compiler=None,
            partition_fn=None,
            decompositions={},
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=False,
        )
        joint = create_joint(prepare_full_output_masks(function), aot_config=config)
        primal_count = len(args)

        def flat_backward(*primals_and_tangents):
            primals = primals_and_tangents[:primal_count]
            tangents = primals_and_tangents[primal_count:]
            # This traces backward with one shared FakeTensorMode, including
            # CPU scheduling carry and CUDA model tensors. Parallel autograd
            # workers race its per-mode kernel flag against thread-local Meta
            # dispatch state. Scope serialization to capture, restoring the
            # caller's policy before any real compiled backward can execute.
            # Checkpoint/save hooks belong to real execution. Letting an outer
            # hook retain these disposable capture saves can replay real inputs
            # into FakeTensorMode during the joint's autograd.grad call.
            with (
                torch.autograd.set_multithreading_enabled(False),
                torch.autograd.graph.saved_tensors_hooks(
                    _capture_saved_tensor_identity, _capture_saved_tensor_identity
                ),
            ):
                outputs, gradients = joint(primals, tangents)
            if len(gradients) != primal_count:
                raise AssertionError(
                    f"Expected {primal_count} input gradients, got {len(gradients)}"
                )
            gradients = [
                torch.zeros_like(primal)
                if isinstance(primal, torch.Tensor) and gradient is None
                else gradient
                for primal, gradient in zip(primals, gradients, strict=True)
            ]
            gradients = clone_aliases(primals_and_tangents, gradients)
            if return_fw_outputs:
                return (*outputs, *gradients)
            return gradients

        return flat_backward

    return create_full_output_backward


def install_scan_compatibility():
    """Install a contract-qualified positional-tangent repair on selection.

    Repeated calls are idempotent. Incompatible interfaces, failing behavior
    qualification or replaced shared helpers are rejected before installation.
    The only assignment into Torch is ``partitioner.create_hop_joint_graph``;
    its cloned function uses private globals, backward helper, and materializer.
    No source files are modified and no source text is executed.
    """
    global _INSTALLED

    if torch.compiler.is_compiling():
        raise FunctionalizationError(
            "Select Dendra scan execution before entering torch.compile."
        )

    with _INSTALL_LOCK:
        if _INSTALLED is not None:
            partitioner, patched, references = _INSTALLED
            if partitioner.create_hop_joint_graph is not patched:
                raise _unsupported("the active scan repair has been replaced")
            _check_references(references)
            return

        modules, references, aot_config_type, create_joint = _validate_scan_contract()
        partitioner = modules["torch._higher_order_ops.partitioner"]
        original = partitioner.create_hop_joint_graph
        private_globals = dict(original.__globals__)
        private_globals["create_bw_fn"] = _make_full_output_backward(
            modules["torch._higher_order_ops.utils"], aot_config_type, create_joint
        )
        private_globals["materialize_as_graph"] = _make_value_preserving_materializer(
            modules["torch._higher_order_ops.utils"]
        )
        patched = FunctionType(
            original.__code__,
            private_globals,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        update_wrapper(patched, original)
        patched.__kwdefaults__ = (
            None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
        )
        _probe_scan_contract(patched)
        _check_references(references)
        # Retain identities of untouched helpers for repeat-call checks. The
        # overridden entry has its own identity check above.
        references = tuple(
            reference
            for reference in references
            if not (reference[0] is partitioner and reference[1] == original.__name__)
        )
        partitioner.create_hop_joint_graph = patched
        _INSTALLED = partitioner, patched, references


__all__ = ["install_scan_compatibility"]
