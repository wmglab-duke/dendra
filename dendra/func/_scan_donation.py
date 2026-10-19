"""Compile-local protection against unproven backward subgraph donation.

The audited Inductor inherits outer AOT saved-tensor donation indices when it
creates a SubgraphLowering.  Those positional indices do not describe the new
child placeholders.  Mark child inputs non-reusable before scheduler codegen;
leave outer donation, explicit authored mutations and ordinary local-buffer
reuse unchanged.  This guard belongs only to Dendra's scan-owned backend.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from torch._inductor.custom_graph_pass import CustomSchedulerPass, get_hash_for_files
from torch._inductor.graph import SubgraphLowering
from torch._inductor.virtualized import V

from ._scan_compat import validate_inductor_scan_contract
from ._types import FunctionalizationError

_MAX_RECEIPTS = 64
_MAX_INPUTS = 4096


def _unsupported(detail):
    return FunctionalizationError(
        "Dendra's Inductor scan input-ownership contract is incompatible: "
        f"{detail}. The selected execution is not silently replaced."
    )


def _validate_contract():
    validate_inductor_scan_contract()


class _DonationGuard:
    def __init__(self, previous=None, *, receipts=None):
        self.previous = previous
        self.receipts = [] if receipts is None else receipts

    def __call__(self, nodes):
        if self.previous is not None:
            nodes = self.previous(nodes)
        _validate_contract()
        graph = V.graph
        if (
            not isinstance(getattr(graph, "is_backward", None), bool)
            or not isinstance(getattr(graph, "graph_input_names", None), (list, tuple))
            or not all(isinstance(name, str) for name in graph.graph_input_names)
            or not callable(
                getattr(getattr(graph, "never_reuse_buffers", None), "update", None)
            )
            or not callable(
                getattr(getattr(graph, "graph_inputs_original", None), "get", None)
            )
            or not hasattr(graph, "name")
            or (graph.name is not None and not isinstance(graph.name, str))
        ):
            raise _unsupported("the lowered graph lacks the input-protection protocol")
        is_subgraph = isinstance(graph, SubgraphLowering)
        backward = bool(graph.is_backward)
        names = list(graph.graph_input_names)
        before = set(graph.never_reuse_buffers)
        applied = is_subgraph and backward
        if applied:
            # Outer donor indices are not an ownership proof for these inputs.
            # This does not disable local-buffer reuse or authored copy_/writes.
            graph.never_reuse_buffers.update(names)
        if len(self.receipts) >= _MAX_RECEIPTS:
            if len(self.receipts) == _MAX_RECEIPTS:
                self.receipts.append(
                    {"status": "truncated", "receipts_truncated": True}
                )
            return nodes
        parent = getattr(graph, "parent", None)
        inputs = []
        for index, name in enumerate(names[:_MAX_INPUTS]):
            original = graph.graph_inputs_original.get(name)
            inputs.append(
                {
                    "index": index,
                    "name": name,
                    "ir_type": None if original is None else type(original).__name__,
                    "was_never_reuse": name in before,
                    "is_never_reuse": name in graph.never_reuse_buffers,
                }
            )
        self.receipts.append(
            {
                "graph_id": hex(id(graph)),
                "graph_name": graph.name,
                "graph_type": type(graph).__name__,
                "parent_id": None if parent is None else hex(id(parent)),
                "parent_name": None
                if parent is None
                else getattr(parent, "name", None),
                "is_backward": backward,
                "is_subgraph": is_subgraph,
                "guard_applied": applied,
                "status": "guarded" if applied else "not_applicable",
                "input_count": len(names),
                "protected_input_count": sum(
                    name in graph.never_reuse_buffers for name in names
                ),
                "inputs": inputs,
                "inputs_truncated": len(names) > _MAX_INPUTS,
                "receipts_truncated": False,
            }
        )
        return nodes

    def diagnostics(self):
        return deepcopy(tuple(self.receipts))


class _ScanDonationPass(_DonationGuard, CustomSchedulerPass):
    def uuid(self):
        previous = None
        if self.previous is not None:
            identity = self.previous.uuid()
            if not identity:
                raise _unsupported("the chained scheduler pass lost its cache identity")
            previous = (
                type(self.previous).__module__,
                type(self.previous).__qualname__,
                identity,
            )
        return (
            "dendra-scan-subgraph-donation-guard-v1",
            get_hash_for_files(
                (
                    __file__,
                    str(Path(__file__).with_name("_scan_inductor.py")),
                    str(Path(__file__).with_name("_scan_compat.py")),
                )
            ),
            previous,
        )


class _UncacheableScanDonationPass(_DonationGuard):
    """A plain callable intentionally retains Torch's FX/AOT cache bypass.

    Wrapping an arbitrary callback in CustomSchedulerPass would mistakenly make
    the compiler accept a cache identity it cannot verify.  The plain callable
    still composes and applies the correctness guard, without patching configs.
    """


def _scan_donation_pass(previous=None, *, receipts=None):
    _validate_contract()
    cacheable = previous is None or (
        isinstance(previous, CustomSchedulerPass) and bool(previous.uuid())
    )
    implementation = _ScanDonationPass if cacheable else _UncacheableScanDonationPass
    return implementation(previous, receipts=receipts)
