"""Composed generic scans preserve small/large recordings and gradients.

These are upstream code-generation correctness checks, not assertions about
Inductor scheduling, allocation counts, or a particular Torch patch release.
"""

from itertools import combinations

import pytest
import torch
from torch._higher_order_ops.scan import scan
from torch._inductor import config

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._scan_compat import install_scan_compatibility
from dendra.func._scan_inductor import _compile_scan

pytestmark = pytest.mark.cpu
_BLOCK = 4
_HALF_STEPS = 8
_STEPS = 2 * _HALF_STEPS


@pytest.fixture(params=[1, 4], ids=["t1", "t4"])
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _program(initial, samples, gain):
    def block(value, block_samples):
        emitted = []
        for offset in range(_BLOCK):
            value = value * gain + block_samples[offset]
            emitted.append(value.clone())
        # Distinct carry/emission storage is scan's ordinary public contract.
        return value.clone(), torch.stack(emitted)

    value, first = scan(
        block,
        initial.clone(),
        samples[:_HALF_STEPS].reshape(_HALF_STEPS // _BLOCK, _BLOCK, -1),
    )
    value, second = scan(
        block,
        value,
        samples[_HALF_STEPS:].reshape(_HALF_STEPS // _BLOCK, _BLOCK, -1),
    )
    # Keep both scan recordings independently owned across later invocations.
    return value.clone(), first.flatten(0, 1), second.flatten(0, 1)


def _reference(initial, samples, gain):
    value, frames = initial, []
    for sample in samples.unbind():
        value = value * gain + sample
        frames.append(value)
    return (
        value,
        torch.stack(frames[:_HALF_STEPS]),
        torch.stack(frames[_HALF_STEPS:]),
    )


def _loss(result):
    final, first, second = result
    return final.square().mean() + first.square().mean() + second.square().mean()


def _inputs(sites, offset):
    # Match the benchmark's float32 site counts at small and larger sizes.
    return (
        torch.linspace(-0.2, 0.3, sites, dtype=torch.float32) + offset,
        torch.linspace(-0.05, 0.07, _STEPS * sites, dtype=torch.float32).reshape(
            _STEPS, sites
        )
        + offset / 10,
        torch.tensor(0.87 + offset / 10, dtype=torch.float32),
    )


@pytest.mark.parametrize("sites", [974, 974 * 8], ids=["small", "large"])
@pytest.mark.slow
def test_generic_composed_scans_keep_owned_recordings_and_all_input_gradients(
    cpu_threads, sites
):
    install_scan_compatibility()
    assert torch.get_num_threads() == cpu_threads
    original_policy = (
        config._pre_fusion_custom_pass,
        config._post_fusion_custom_pass,
        config.cpp.min_chunk_size,
    )
    compiled = _compile_scan(
        _program,
        {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "options": {
                "cpp_wrapper": True,
                "compile_threads": 1,
                "fx_graph_cache": False,
            },
        },
    )
    earlier_outputs = None
    earlier_snapshots = None
    for offset in (0.0, 0.03):
        values = _inputs(sites, offset)
        actual = tuple(value.clone().requires_grad_() for value in values)
        expected = tuple(value.clone().requires_grad_() for value in values)
        saved = tuple(value.detach().clone() for value in actual)
        versions = tuple(value._version for value in actual)
        # Capturing compiler output is scoped to this invocation and does not
        # reset Dynamo, replace a scheduler, or alter generated computations.
        with torch_compiler_warning_context():
            output = compiled(*actual)
            reference = _reference(*expected)
            gradients = torch.autograd.grad(_loss(output), actual)
            wanted = torch.autograd.grad(_loss(reference), expected)
        assert output[0].shape == (sites,)
        assert output[1].shape == output[2].shape == (_HALF_STEPS, sites)
        torch.testing.assert_close(output, reference, rtol=3e-5, atol=3e-6)
        for gradient, correct in zip(gradients, wanted, strict=True):
            assert torch.isfinite(gradient).all()
            assert torch.count_nonzero(correct) > 0
            torch.testing.assert_close(gradient, correct, rtol=3e-5, atol=3e-7)
        for current, before, version in zip(actual, saved, versions, strict=True):
            torch.testing.assert_close(current, before, rtol=0, atol=0)
            assert current._version == version
        for left, right in combinations(output, 2):
            assert not torch._C._is_alias_of(left, right)
        for result in output:
            assert all(not torch._C._is_alias_of(result, value) for value in actual)
        if earlier_outputs is not None:
            torch.testing.assert_close(
                earlier_outputs, earlier_snapshots, rtol=0, atol=0
            )
        earlier_outputs = output
        earlier_snapshots = tuple(value.detach().clone() for value in output)
    assert (
        config._pre_fusion_custom_pass,
        config._post_fusion_custom_pass,
        config.cpp.min_chunk_size,
    ) == original_policy
