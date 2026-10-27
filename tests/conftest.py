"""Shared test classification for required and optional execution lanes."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

import pytest

# Import Dendra once before test-module collection so process-wide runtime
# initialization is deterministic. TorchInductor cache configuration is opt-in.
import dendra  # noqa: F401
from dendra._bootstrap import reset_torch_compiler

_MPS_REASON = re.compile(r"\b(?:mps|metal)\b", re.IGNORECASE)

_NEURON_MODULES = {
    "tests/test_io_from_neuron_topology.py",
    "tests/simtests/test_against_neuron.py",
}
_STOCHASTIC_TOKENS = (
    "noise_positive",
    "poisson",
    "random",
    "seed",
    "stochastic",
)
# PyTorch's supported JIT APIs emit deprecations, and Dynamo inspects .grad
# on nonleaf operands while tracing. Match only these known upstream messages
# at their source modules; user warnings and other compiler warnings remain
# errors. Per-item filters override the required lane's ``-W error``; config
# filters do not.
_UPSTREAM_COMPILER_WARNING_FILTERS = (
    tuple(
        r"ignore:^`?torch\.jit\.(script|script_method)`? is deprecated(\.|$).*:"
        + category
        + r":^torch\.jit\._script$"
        for category in ("FutureWarning", "DeprecationWarning")
    )
    + tuple(
        r"ignore:^`?torch\.jit\.trace`? is deprecated(\.|$).*:"
        + category
        + r":^torch\.jit\._trace$"
        for category in ("FutureWarning", "DeprecationWarning")
    )
    + (
        (
            r"ignore:^The \.grad attribute of a Tensor that is not a leaf Tensor "
            r"is being accessed\. Its \.grad attribute won't be populated during "
            r"autograd\.backward\(\)\. If you indeed want the \.grad field to be "
            r"populated for a non-leaf Tensor, use \.retain_grad\(\) on the non-leaf "
            r"Tensor\. If you access the non-leaf Tensor by mistake, make sure you "
            r"access the leaf Tensor instead\. See github\.com/pytorch/pytorch/pull/30531 "
            r"for more information\.( \(Triggered internally at .*\))?$:"
            r"UserWarning:^torch\.(_dynamo\.variables\.builder|_subclasses\.meta_utils)$"
        ),
    )
)


@pytest.fixture(autouse=True)
def isolated_torch_compiler_cache():
    """Keep independent tests from sharing Torch's specialization budget.

    Do not reset inside a test: lifecycle and cache-reuse assertions must see
    every compilation performed by that case under the ordinary cache limits.
    The helper uses public ``torch.compiler.reset`` with known-warning handling.
    """
    reset_torch_compiler()
    try:
        yield
    finally:
        reset_torch_compiler()


def _parameter_uses_cuda(item: pytest.Item) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    return any(
        str(value).lower().startswith("cuda") for value in callspec.params.values()
    )


def _skipif_requires_cuda(item: pytest.Item) -> bool:
    """Recognize accelerator guards without inferring intent from a test name."""
    for marker in item.iter_markers(name="skipif"):
        reason = str(marker.kwargs.get("reason", "")).casefold()
        if "cuda" in reason or ("gpu" in reason and not _MPS_REASON.search(reason)):
            return True
    return False


def _requires_mps(item: pytest.Item, authored_cpu: set) -> bool:
    if item.get_closest_marker("mps") is not None:
        return True
    for marker in item.iter_markers(name="skipif"):
        if _MPS_REASON.search(str(marker.kwargs.get("reason", ""))):
            return True
    callspec = getattr(item, "callspec", None)
    return (
        item not in authored_cpu
        and callspec is not None
        and any(
            str(value).lower().startswith("mps") for value in callspec.params.values()
        )
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign every collected test to a required or optional backend lane.

    Tests may carry more than one backend marker (for example the CUDA NEURON
    comparison), but only tests requiring no accelerator receive ``cpu``.
    Parameterized ``device='cuda'`` cases are classified independently from
    their CPU siblings unless an authored ``cpu`` marker declares that only
    device metadata is inspected. Explicit accelerator markers and hardware guards
    still identify accelerator requirements, including within CPU-marked
    modules or classes.
    """

    authored_cpu = {
        item for item in items if item.get_closest_marker("cpu") is not None
    }
    # Pytest inherits module/class markers even when a case declares another
    # backend. Restore CPU markers per case so explicit CUDA requirements do
    # not also satisfy the required lane's ``cpu`` alternative.
    cpu_owners = {
        owner
        for item in authored_cpu
        for owner, _marker in item.iter_markers_with_node("cpu")
    }
    for owner in cpu_owners:
        owner.own_markers[:] = [
            marker for marker in owner.own_markers if marker.name != "cpu"
        ]

    for item in items:
        for warning_filter in _UPSTREAM_COMPILER_WARNING_FILTERS:
            item.add_marker(pytest.mark.filterwarnings(warning_filter))
        module_path = item.nodeid.split("::", 1)[0]
        normalized_path = str(PurePosixPath(module_path))
        nodeid_lower = item.nodeid.lower()

        needs_neuron = (
            normalized_path in _NEURON_MODULES
            or item.get_closest_marker("neuron") is not None
        )
        needs_cuda = (
            (item not in authored_cpu and _parameter_uses_cuda(item))
            or _skipif_requires_cuda(item)
            or item.get_closest_marker("cuda") is not None
        )
        needs_mps = _requires_mps(item, authored_cpu)

        if needs_neuron:
            item.add_marker(pytest.mark.neuron)
        if needs_cuda:
            item.add_marker(pytest.mark.cuda)
        if needs_mps:
            item.add_marker(pytest.mark.mps)
        if normalized_path.startswith("tests/simtests/"):
            item.add_marker(pytest.mark.slow)
        if any(token in nodeid_lower for token in _STOCHASTIC_TOKENS):
            item.add_marker(pytest.mark.stochastic)

        if (
            not needs_cuda
            and not needs_mps
            and (item in authored_cpu or not needs_neuron)
        ):
            item.add_marker(pytest.mark.cpu)
