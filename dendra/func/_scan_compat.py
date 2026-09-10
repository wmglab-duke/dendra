"""Explicit compatibility activation for Dendra's experimental scan executor.

PyTorch 2.14's HOP partitioner supplies one cotangent slot for every output,
including integer and Boolean outputs. Its shared backward helper otherwise
filters those outputs, producing mismatched positional signatures. The repair
below gives only the partitioner a private backward helper with the full mask.
Shared utility functions and their existing callers are never replaced.

Activation is persistent and process-wide at that partitioner entry point:
backward and checkpoint replay can run after the selecting forward returns.
Importing this module performs no activation and imports no private Torch HOP.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import threading
from functools import update_wrapper
from pathlib import Path
from types import FunctionType

import torch

from ._types import FunctionalizationError

_SUPPORTED_VERSION = "2.14.0"
_SOURCE_HASHES = {
    "torch._higher_order_ops.utils": (
        "00b426146c9974b8c536703914fa0d4773070ee259b2b05cebd988c43f0c39d8"
    ),
    "torch._higher_order_ops.partitioner": (
        "04da0de4820305b54a679335ec13934942b4f444ed791e0b738ef34b4b7584ea"
    ),
    "torch._functorch._aot_autograd.graph_capture_wrappers": (
        "229d5d5575539f27b199d835d18314e14e99e98c4755d46724dedbb12683d8c8"
    ),
    "torch._higher_order_ops.scan": (
        "6bb980d8ceecf0455a6f45400105fc1d3e524c6c7c1f93af7543aea6ad92d87b"
    ),
}
_FUNCTION_HASHES = {
    ("torch._higher_order_ops.utils", "create_bw_fn"): (
        "06d2246919c415d678de23a62ce7bebf027b06d261986f72eb6c9fceac035c04"
    ),
    (
        "torch._higher_order_ops.utils",
        "prepare_fw_with_masks_all_requires_grad",
    ): "0f86a3004800d832293a7943b1d74237e47e644b2ca59d0772a87129707335e2",
    ("torch._higher_order_ops.utils", "_clone_aliasing_output"): (
        "2008efe666f59d0595914684cf4a00b43c389aa849a529bed56b7c6640b8257f"
    ),
    ("torch._higher_order_ops.partitioner", "create_hop_joint_graph"): (
        "b408b50d910adcc5864d73925a4b56034e5be264c5b4417c9e355529e6733cd9"
    ),
    (
        "torch._functorch._aot_autograd.graph_capture_wrappers",
        "create_joint",
    ): "317f67a953b4f6260f0daed8c9270c8d194d1b73939ea20553d1dfb6e4c19846",
    ("torch._higher_order_ops.scan", "scan_autograd"): (
        "6697a243af15ecfd7a01cc1ae85c25817fc27e6b5f9781b95b57040f50d1517a"
    ),
}
_INSTALL_LOCK = threading.Lock()
_INSTALLED = None


def _unsupported(details):
    return FunctionalizationError(
        "Dendra scan execution requires the verified PyTorch 2.14.0 "
        f"implementation; {details}. Use the default compiled execution or "
        "a supported Torch installation."
    )


def _validate_sources():
    modules = {}
    for name, expected in _SOURCE_HASHES.items():
        try:
            module = importlib.import_module(name)
            source = Path(inspect.getsourcefile(module)).read_bytes()
        except (ImportError, OSError, TypeError) as error:
            raise _unsupported(f"cannot inspect {name}: {error}") from error
        actual = hashlib.sha256(source).hexdigest()
        if actual != expected:
            raise _unsupported(f"source fingerprint differs for {name}")
        modules[name] = module

    references = []
    for (name, attribute), expected in _FUNCTION_HASHES.items():
        module = modules[name]
        function = getattr(module, attribute, None)
        if (
            not isinstance(function, FunctionType)
            or function.__globals__ is not module.__dict__
        ):
            raise _unsupported(f"{name}.{attribute} has already been replaced")
        try:
            actual = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        except (OSError, TypeError) as error:
            raise _unsupported(f"cannot inspect {name}.{attribute}") from error
        if actual != expected:
            raise _unsupported(f"callable fingerprint differs for {name}.{attribute}")
        references.append((module, attribute, function))

    utils = modules["torch._higher_order_ops.utils"]
    partitioner = modules["torch._higher_order_ops.partitioner"]
    if partitioner.create_bw_fn is not utils.create_bw_fn:
        raise _unsupported("the partitioner's backward helper has already changed")
    # The implementation below uses exactly this AOT constructor/entry point.
    aot = importlib.import_module("torch._functorch.aot_autograd")
    wrappers = modules["torch._functorch._aot_autograd.graph_capture_wrappers"]
    if aot.create_joint is not wrappers.create_joint:
        raise _unsupported("the AOT joint entry point has already changed")
    references.append((aot, "create_joint", aot.create_joint))
    return modules, tuple(references), aot.AOTConfig, aot.create_joint


def _make_full_output_backward(utils, aot_config_type, create_joint):
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
    """Install the verified positional-tangent repair after explicit selection.

    Repeated calls are idempotent. Unknown Torch versions, changed installed
    sources, or replaced shared helpers are rejected before installation.
    The only assignment into Torch is ``partitioner.create_hop_joint_graph``;
    its cloned function uses private globals and a private backward helper.
    No source files are modified and no source text is executed.
    """
    global _INSTALLED

    version = str(torch.__version__)
    if version.split("+", 1)[0] != _SUPPORTED_VERSION:
        raise _unsupported(f"found Torch {version}")
    if torch.compiler.is_compiling():
        raise FunctionalizationError(
            "Select Dendra scan execution before entering torch.compile."
        )

    with _INSTALL_LOCK:
        if _INSTALLED is not None:
            partitioner, patched, references = _INSTALLED
            if partitioner.create_hop_joint_graph is not patched:
                raise _unsupported("the active scan repair has been replaced")
            for module, name, original in references:
                if getattr(module, name, None) is not original:
                    raise _unsupported(
                        f"the active helper {module.__name__}.{name} changed"
                    )
            return

        modules, references, aot_config_type, create_joint = _validate_sources()
        partitioner = modules["torch._higher_order_ops.partitioner"]
        original = partitioner.create_hop_joint_graph
        private_globals = dict(original.__globals__)
        private_globals["create_bw_fn"] = _make_full_output_backward(
            modules["torch._higher_order_ops.utils"], aot_config_type, create_joint
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
