"""Selection-policy tests for material diffusion linear solvers."""

import sys
from functools import partial
from types import ModuleType, SimpleNamespace

import pytest
import torch

from dendra.models.integrators.tridiag import pcr_solve_parallel_t
from dendra.models.mechanisms import _spatial as spatial


def _solver(name):
    def solve(*args, **kwargs):
        return name, args, kwargs

    return solve


@pytest.mark.parametrize("name", ["dense", "debug", "torch", "linalg"])
def test_dense_tridiagonal_aliases_never_require_an_accelerated_solver(name):
    assert spatial.select_tridiagonal_solver(name, "cpu") == (None, "dense")


def test_explicit_pcr_selection_is_device_specific(monkeypatch):
    cpu = _solver("cpu")
    cuda = _solver("cuda")
    mps = _solver("mps")
    monkeypatch.setattr(spatial, "pcr_solve_t", cpu)
    monkeypatch.setattr(spatial, "pcr_solve_cuda_t", cuda)
    monkeypatch.setattr(spatial, "pcr_solve_parallel_t", mps)

    assert spatial.select_tridiagonal_solver("pcr", torch.device("cpu")) == (
        cpu,
        "pcr_cpu",
    )
    assert spatial.select_tridiagonal_solver("pcr", "cuda") == (cuda, "pcr_cuda")
    assert spatial.select_tridiagonal_solver("pcr", "mps") == (mps, "pcr_mps")
    assert spatial.select_tridiagonal_solver("pcr", "mps:0") == (mps, "pcr_mps")

    monkeypatch.setattr(spatial, "pcr_solve_t", None)
    with pytest.raises(ImportError, match="integrators.tridiag.pcr_solve_t"):
        spatial.select_tridiagonal_solver("pcr", "cpu")
    monkeypatch.setattr(spatial, "pcr_solve_cuda_t", None)
    with pytest.raises(ImportError, match="integrators.triton.pcr_solve_cuda_t"):
        spatial.select_tridiagonal_solver("pcr", "cuda")
    monkeypatch.setattr(spatial, "pcr_solve_parallel_t", None)
    with pytest.raises(ImportError, match="tridiag.pcr_solve_parallel_t"):
        spatial.select_tridiagonal_solver("pcr", "mps")
    assert spatial.select_tridiagonal_solver("auto", "mps") == (None, "dense")


def test_spd_uses_cpu_extension_and_warns_before_cuda_fallback(monkeypatch):
    solve_tri_spd = _solver("spd")
    extension = ModuleType("dendra_solvers")
    extension.solve_tri_spd = solve_tri_spd
    monkeypatch.setitem(sys.modules, "dendra_solvers", extension)

    assert spatial.select_tridiagonal_solver("spd", "cpu") == (
        solve_tri_spd,
        "spd_cpu",
    )

    thomas = _solver("thomas")
    monkeypatch.setattr(spatial, "thomas_solve_cuda_t", thomas)
    with pytest.warns(UserWarning, match="not implemented on CUDA"):
        assert spatial.select_tridiagonal_solver("spd", "cuda") == (
            thomas,
            "thomas_cuda",
        )

    mps = _solver("mps")
    monkeypatch.setattr(spatial, "pcr_solve_parallel_t", mps)
    with pytest.warns(UserWarning, match="not implemented on MPS"):
        assert spatial.select_tridiagonal_solver("spd", "mps") == (
            mps,
            "pcr_mps",
        )


def test_missing_cpu_spd_extension_falls_back_through_cpu_policy(monkeypatch):
    incomplete_extension = ModuleType("dendra_solvers")
    monkeypatch.setitem(sys.modules, "dendra_solvers", incomplete_extension)
    fallback = _solver("fallback")
    monkeypatch.setattr(spatial, "DENDRA_SOLVERS_AVAILABLE", False)
    monkeypatch.setattr(spatial, "pcr_solve_t", fallback)

    with pytest.warns(UserWarning) as recorded:
        selected = spatial.select_tridiagonal_solver("spd", "cpu")
    assert selected == (fallback, "pcr_cpu")
    assert [str(item.message) for item in recorded] == [
        "Material diffusion solver='spd' requested on CPU but "
        "dendra_solvers.solve_tri_spd is unavailable; falling back to Thomas/PCR.",
        "Using material diffusion on CPU without dendra-solvers installed; "
        "falling back to the PyTorch PCR/Thomas solver. Install the native "
        "package with `python -m pip install --upgrade "
        '--only-binary=dendra-solvers "dendra-solvers>=0.3.1"`.',
    ]


def test_cuda_thomas_policy_prefers_thomas_then_pcr_then_dense(monkeypatch):
    thomas = _solver("thomas")
    pcr = _solver("pcr")
    monkeypatch.setattr(spatial, "thomas_solve_cuda_t", thomas)
    monkeypatch.setattr(spatial, "pcr_solve_cuda_t", pcr)
    assert spatial.select_tridiagonal_solver("auto", "cuda") == (
        thomas,
        "thomas_cuda",
    )

    monkeypatch.setattr(spatial, "thomas_solve_cuda_t", None)
    with pytest.warns(UserWarning, match="falling back to CUDA PCR"):
        assert spatial.select_tridiagonal_solver("auto", "cuda") == (
            pcr,
            "pcr_cuda",
        )

    monkeypatch.setattr(spatial, "pcr_solve_cuda_t", None)
    assert spatial.select_tridiagonal_solver("auto", "cuda") == (None, "dense")
    with pytest.raises(ImportError, match="solver='thomas' on CUDA"):
        spatial.select_tridiagonal_solver("thomas", "cuda")


def test_cpu_thomas_policy_covers_extension_fallback_and_dense(monkeypatch):
    thomas = _solver("thomas")
    monkeypatch.setattr(
        spatial.torch.ops,
        "dendra_solvers",
        SimpleNamespace(thomas_solve_t=thomas),
    )
    monkeypatch.setattr(spatial, "DENDRA_SOLVERS_AVAILABLE", True)
    solve, name = spatial.select_tridiagonal_solver("auto", "cpu")
    assert solve is thomas
    assert name == "thomas_cpu"

    fallback = _solver("fallback")
    monkeypatch.setattr(spatial, "DENDRA_SOLVERS_AVAILABLE", False)
    monkeypatch.setattr(spatial, "pcr_solve_t", fallback)
    with pytest.warns(UserWarning, match="without dendra-solvers"):
        assert spatial.select_tridiagonal_solver("thomas", "cpu") == (
            fallback,
            "pcr_cpu",
        )

    monkeypatch.setattr(spatial, "pcr_solve_t", None)
    assert spatial.select_tridiagonal_solver("auto", "cpu") == (None, "dense")


def test_unknown_tridiagonal_solver_warns_and_mps_auto_uses_parallel_pcr(monkeypatch):
    monkeypatch.setattr(spatial, "DENDRA_SOLVERS_AVAILABLE", False)
    monkeypatch.setattr(spatial, "pcr_solve_t", None)
    mps = _solver("mps")
    monkeypatch.setattr(spatial, "pcr_solve_parallel_t", mps)
    with pytest.warns(UserWarning, match="Unknown material tridiagonal solver"):
        assert spatial.select_tridiagonal_solver("surprise", "cpu") == (
            None,
            "dense",
        )
    assert spatial.select_tridiagonal_solver("auto", "mps") == (mps, "pcr_mps")
    assert spatial.select_tridiagonal_solver("thomas", "mps") == (mps, "pcr_mps")


def _dense_tridiagonal(a, b, c):
    matrix = torch.diag_embed(b)
    rows = torch.arange(b.shape[-1] - 1, device=b.device)
    matrix[..., rows + 1, rows] = a
    matrix[..., rows, rows + 1] = c
    return matrix


@pytest.mark.parametrize("length", [1, 2, 3, 5, 11, 17, 32, 33, 101])
def test_parallel_pcr_matches_dense_for_arbitrary_diffusion_system_sizes(length):
    """Cover singleton, prime, and adjacent power-of-two chain lengths."""
    generator = torch.Generator().manual_seed(1000 + length)
    leading_shape = (2, 3)
    edge = torch.rand(
        leading_shape + (max(length - 1, 0),),
        generator=generator,
        dtype=torch.float64,
    )
    mass = 0.25 + torch.rand(
        leading_shape + (length,), generator=generator, dtype=torch.float64
    )
    boundary = torch.zeros(leading_shape + (1,), dtype=torch.float64)
    a = -edge
    c = -edge
    b = mass + torch.cat((boundary, edge), dim=-1) + torch.cat((edge, boundary), dim=-1)
    rhs = torch.randn(
        leading_shape + (length,), generator=generator, dtype=torch.float64
    )

    expected = torch.linalg.solve(
        _dense_tridiagonal(a, b, c), rhs.unsqueeze(-1)
    ).squeeze(-1)
    actual = pcr_solve_parallel_t(a, b, c, rhs)

    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def test_parallel_pcr_is_autograd_safe_for_all_bands_and_rhs():
    a = torch.tensor([[-0.2, 0.1, -0.3, 0.2]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(
        [[2.0, 2.5, 3.0, 2.25, 1.75]], dtype=torch.float64, requires_grad=True
    )
    c = torch.tensor([[0.3, -0.1, 0.15, -0.2]], dtype=torch.float64, requires_grad=True)
    rhs = torch.tensor(
        [[1.0, -2.0, 0.5, 1.25, -0.75]],
        dtype=torch.float64,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(
        pcr_solve_parallel_t,
        (a, b, c, rhs),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_parallel_pcr_preserves_constant_field_for_tjs_sized_diffusion_system():
    length = 1101
    generator = torch.Generator().manual_seed(20260821)
    edge = torch.rand((5, length - 1), generator=generator, dtype=torch.float32)
    mass = 0.5 + torch.rand((5, length), generator=generator, dtype=torch.float32)
    boundary = torch.zeros((5, 1), dtype=torch.float32)
    a = -edge
    c = -edge
    b = mass + torch.cat((boundary, edge), dim=-1) + torch.cat((edge, boundary), dim=-1)
    expected = torch.arange(1, 6, dtype=torch.float32).unsqueeze(-1).expand(-1, length)
    rhs = mass * expected

    actual = pcr_solve_parallel_t(a, b, c, rhs)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "name", ["dense", "debug", "torch", "linalg", "dhs", "hines", "tree", "custom"]
)
def test_tree_solver_aliases_follow_dense_or_auto_policy(monkeypatch, name):
    monkeypatch.setattr(spatial, "DENDRA_SOLVERS_AVAILABLE", False)
    if name in {"dense", "debug", "torch", "linalg"}:
        assert spatial._select_dhs_solver(name, "cpu", threads=64) == (None, "dense")
    else:
        with pytest.warns(UserWarning, match="without dendra-solvers"):
            assert spatial._select_dhs_solver(name, "cpu", threads=64) == (
                None,
                "dense",
            )


def test_tree_cuda_policy_wraps_threads_and_handles_missing_solver(monkeypatch):
    solver = _solver("dhs")
    monkeypatch.setattr(spatial, "dhs_solve_cuda", solver)
    selected, name = spatial._select_dhs_solver("auto", "cuda", threads=128)
    assert isinstance(selected, partial)
    assert selected.func is solver
    assert selected.keywords == {"threads": 128}
    assert name == "dhs_cuda"

    monkeypatch.setattr(spatial, "dhs_solve_cuda", None)
    assert spatial._select_dhs_solver("auto", "cuda", threads=128) == (
        None,
        "dense",
    )
    with pytest.raises(ImportError, match="Tree material diffusion on CUDA"):
        spatial._select_dhs_solver("thomas", "cuda", threads=128)


def test_tree_cpu_policy_covers_extension_explicit_failure_and_fallback(monkeypatch):
    dhs = _solver("dhs")
    monkeypatch.setattr(
        spatial.torch.ops,
        "dendra_solvers",
        SimpleNamespace(dhs_solve=dhs),
    )
    monkeypatch.setattr(spatial, "DENDRA_SOLVERS_AVAILABLE", True)
    solve, name = spatial._select_dhs_solver("thomas", "cpu", threads=32)
    assert solve is dhs
    assert name == "dhs_cpu"

    monkeypatch.setattr(spatial, "DENDRA_SOLVERS_AVAILABLE", False)
    with pytest.raises(ImportError, match="requires the dendra-solvers package"):
        spatial._select_dhs_solver("thomas", "cpu", threads=32)
    with pytest.warns(UserWarning, match="falling back to dense"):
        assert spatial._select_dhs_solver("auto", "cpu", threads=32) == (
            None,
            "dense",
        )


def test_unknown_tree_solver_warns_and_other_devices_use_dense(monkeypatch):
    monkeypatch.setattr(spatial, "DENDRA_SOLVERS_AVAILABLE", False)
    with pytest.warns(UserWarning) as recorded:
        assert spatial._select_dhs_solver("surprise", "cpu", threads=16) == (
            None,
            "dense",
        )
    assert len(recorded) == 2
    assert "Unknown tree material solver" in str(recorded[0].message)
    assert "falling back to dense" in str(recorded[1].message)
    assert spatial._select_dhs_solver("auto", "mps", threads=16) == (None, "dense")
