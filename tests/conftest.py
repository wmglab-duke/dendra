"""Shared test classification for required and optional execution lanes."""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest

# Dendra configures TorchInductor during package import. Import it before test
# modules have a chance to import torch so collection itself exercises the
# supported initialization order and warnings can be treated as errors.
import dendra  # noqa: F401

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
        if "cuda" in reason or "gpu" in reason:
            return True
    return False


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign every collected test to a required or optional backend lane.

    Tests may carry more than one backend marker (for example the CUDA NEURON
    comparison), but only tests requiring neither receive the ``cpu`` marker.
    Parameterized ``device='cuda'`` cases are classified independently from
    their CPU siblings.
    """

    for item in items:
        module_path = item.nodeid.split("::", 1)[0]
        normalized_path = str(PurePosixPath(module_path))
        nodeid_lower = item.nodeid.lower()

        needs_neuron = (
            normalized_path in _NEURON_MODULES
            or item.get_closest_marker("neuron") is not None
        )
        needs_cuda = (
            _parameter_uses_cuda(item)
            or _skipif_requires_cuda(item)
            or item.get_closest_marker("cuda") is not None
        )

        if needs_neuron:
            item.add_marker(pytest.mark.neuron)
        if needs_cuda:
            item.add_marker(pytest.mark.cuda)
        if normalized_path.startswith("tests/simtests/"):
            item.add_marker(pytest.mark.slow)
        if any(token in nodeid_lower for token in _STOCHASTIC_TOKENS):
            item.add_marker(pytest.mark.stochastic)

        if not needs_neuron and not needs_cuda:
            item.add_marker(pytest.mark.cpu)
