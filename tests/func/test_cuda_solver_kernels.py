"""Real CUDA kernel acceptance against independent dense CPU algebra.

Unlike test_cuda_functional_reference, this module never injects launchers.
It must skip on CPU-only hosts and is not evidence of GPU performance.
"""

from __future__ import annotations

import pytest
import torch

from dendra.models.integrators import _cuda_functional as facade
from tests.func.test_cuda_functional_reference import FAMILIES, _case, _values

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _cuda_solve(family, metadata):
    """Bind actual CUDA topology, never the CPU case's solver closure."""
    if family == "thomas":
        return facade.thomas_solve
    if family == "pcr":
        return facade.pcr_solve
    if family == "block_thomas":
        return facade.block_thomas_solve
    if family == "dhs":

        def solve(main, edge, rhs):
            return facade.dhs_solve(main, edge, rhs, *metadata, threads=32)

    elif family == "dhs_bt":

        def solve(main, edge, rhs):
            return facade.dhs_bt_solve(main, edge, rhs, *metadata, threads=32)

    else:

        def solve(main, edge, rhs):
            return facade.dhs_multi_solve(
                main,
                edge,
                rhs,
                *metadata,
                K_stride=5,
                L_max=2,
                threads=16,
                grid_x=2,
            )

    return solve


def _close(actual, expected, *, dtype):
    torch.testing.assert_close(
        actual.detach().cpu(),
        expected.detach().cpu(),
        rtol=5e-4 if dtype == torch.float32 else 1e-8,
        atol=5e-6 if dtype == torch.float32 else 1e-10,
    )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_real_cuda_solver_primal_all_gradients_jvp_vmap_and_strided_topology(
    family, dtype
):
    case = _case(family)
    cpu = tuple(value.to(dtype=dtype).clone().requires_grad_() for value in case.values)
    gpu = tuple(value.detach().to(device="cuda").requires_grad_() for value in cpu)
    metadata = tuple(value.to(device="cuda") for value in case.metadata)
    solve = _cuda_solve(family, metadata)
    snapshots = tuple(
        (value.detach().cpu().clone(), value._version) for value in (*gpu, *metadata)
    )

    actual, expected = solve(*gpu), case.reference(*cpu)
    _close(actual, expected, dtype=dtype)
    weights = _values(expected.shape, start=0.2).to(dtype=dtype)
    gradients = torch.autograd.grad(
        (actual.square() * weights.to(device="cuda")).sum(), gpu
    )
    reference_gradients = torch.autograd.grad((expected.square() * weights).sum(), cpu)
    assert len(gradients) == len(gpu)
    for gradient, reference in zip(gradients, reference_gradients, strict=True):
        _close(gradient, reference, dtype=dtype)
        assert torch.isfinite(gradient).all()

    tangents = tuple(_values(value.shape, start=0.02).to(dtype=dtype) for value in cpu)
    cuda_jvp = torch.func.jvp(
        solve,
        tuple(value.detach() for value in gpu),
        tuple(value.to(device="cuda") for value in tangents),
    )
    cpu_jvp = torch.func.jvp(
        case.reference, tuple(value.detach() for value in cpu), tangents
    )
    for value, reference in zip(cuda_jvp, cpu_jvp, strict=True):
        _close(value, reference, dtype=dtype)

    sources = torch.stack(
        (gpu[-1].detach(), gpu[-1].detach() * 0.8, gpu[-1].detach() * 1.2)
    )

    def mapped(source):
        return solve(*(value.detach() for value in gpu[:-1]), source)

    mapped_actual = torch.vmap(mapped)(sources)
    mapped_expected = torch.stack(
        [
            case.reference(*(value.detach() for value in cpu[:-1]), source.cpu())
            for source in sources
        ]
    )
    _close(mapped_actual, mapped_expected, dtype=dtype)

    if metadata:
        strided = tuple(
            torch.stack((value, torch.zeros_like(value)), dim=-1).reshape(-1)[::2]
            for value in metadata
        )
        assert all(
            value.stride() == (2,) and not value.is_contiguous() for value in strided
        )
        _close(
            _cuda_solve(family, strided)(*(value.detach() for value in gpu)),
            expected,
            dtype=dtype,
        )
    if family in ("dhs", "dhs_bt"):
        assert torch.count_nonzero(gpu[1].detach()[:, 2]) == 0
        assert torch.count_nonzero(gradients[1][:, 2])
    if family == "multi":
        assert torch.count_nonzero(actual[:2, 3:]) == 0
        assert torch.count_nonzero(actual[2, 2:]) == 0
        for gradient in gradients:
            assert torch.count_nonzero(gradient[:2, 3:]) == 0
            assert torch.count_nonzero(gradient[2, 2:]) == 0
    for value, (snapshot, version) in zip((*gpu, *metadata), snapshots, strict=True):
        assert value._version == version
        torch.testing.assert_close(value.detach().cpu(), snapshot, rtol=0, atol=0)
