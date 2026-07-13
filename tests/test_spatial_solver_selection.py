"""Selection-policy tests for material diffusion linear solvers."""

import sys
from functools import partial
from types import ModuleType, SimpleNamespace

import pytest
import torch

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
    monkeypatch.setattr(spatial, "pcr_solve_t", cpu)
    monkeypatch.setattr(spatial, "pcr_solve_cuda_t", cuda)

    assert spatial.select_tridiagonal_solver("pcr", torch.device("cpu")) == (
        cpu,
        "pcr_cpu",
    )
    assert spatial.select_tridiagonal_solver("pcr", "cuda") == (cuda, "pcr_cuda")

    monkeypatch.setattr(spatial, "pcr_solve_t", None)
    with pytest.raises(ImportError, match="integrators.tridiag.pcr_solve_t"):
        spatial.select_tridiagonal_solver("pcr", "cpu")
    monkeypatch.setattr(spatial, "pcr_solve_cuda_t", None)
    with pytest.raises(ImportError, match="integrators.triton.pcr_solve_cuda_t"):
        spatial.select_tridiagonal_solver("pcr", "cuda")


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
        "Using material diffusion on CPU without dendra_solvers installed; "
        "falling back to PCR/Thomas torch solver.",
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
    with pytest.warns(UserWarning, match="without dendra_solvers"):
        assert spatial.select_tridiagonal_solver("thomas", "cpu") == (
            fallback,
            "pcr_cpu",
        )

    monkeypatch.setattr(spatial, "pcr_solve_t", None)
    assert spatial.select_tridiagonal_solver("auto", "cpu") == (None, "dense")


def test_unknown_tridiagonal_solver_warns_and_other_devices_use_dense(monkeypatch):
    monkeypatch.setattr(spatial, "DENDRA_SOLVERS_AVAILABLE", False)
    monkeypatch.setattr(spatial, "pcr_solve_t", None)
    with pytest.warns(UserWarning, match="Unknown material tridiagonal solver"):
        assert spatial.select_tridiagonal_solver("surprise", "cpu") == (
            None,
            "dense",
        )
    assert spatial.select_tridiagonal_solver("auto", "mps") == (None, "dense")


@pytest.mark.parametrize(
    "name", ["dense", "debug", "torch", "linalg", "dhs", "hines", "tree", "custom"]
)
def test_tree_solver_aliases_follow_dense_or_auto_policy(monkeypatch, name):
    monkeypatch.setattr(spatial, "DENDRA_SOLVERS_AVAILABLE", False)
    if name in {"dense", "debug", "torch", "linalg"}:
        assert spatial._select_dhs_solver(name, "cpu", threads=64) == (None, "dense")
    else:
        with pytest.warns(UserWarning, match="without dendra_solvers"):
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
    with pytest.raises(ImportError, match="requires dendra_solvers.dhs_solve"):
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
