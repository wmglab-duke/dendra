"""MPS solver capability selection is testable without allocating an accelerator."""

from types import SimpleNamespace

import pytest
import torch

from dendra.models.integrators import implicit, tree, tree_bt
from dendra.models.integrators._native import _mps_solver_available
from dendra.models.integrators.tridiag import pcr_solve_parallel_t
from dendra.models.integrators.tridiag.block import block_pcr_solve_t

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("method", ("thomas", "pcr"))
@pytest.mark.parametrize("native", (False, True))
def test_scalar_mps_selection_prefers_native_and_keeps_portable_fallback(
    monkeypatch, method, native
):
    chosen = object()
    monkeypatch.setattr(implicit, "DENDRA_SOLVERS_AVAILABLE", True)
    monkeypatch.setattr(implicit, "_mps_solver_available", lambda _module: native)
    monkeypatch.setattr(
        torch.ops, "dendra_solvers", SimpleNamespace(**{f"{method}_solve_t": chosen})
    )
    integrator = SimpleNamespace(method=method, _solve=None)
    implicit._bwd_euler_ub._select_solver(integrator, "mps:0")
    assert integrator._solve is (chosen if native else pcr_solve_parallel_t)


@pytest.mark.parametrize("native", (False, True))
def test_block_mps_selection_prefers_native_and_keeps_portable_fallback(
    monkeypatch, native
):
    chosen = object()
    monkeypatch.setattr(implicit, "DENDRA_SOLVERS_AVAILABLE", True)
    monkeypatch.setattr(implicit, "_mps_solver_available", lambda _module: native)
    monkeypatch.setattr(torch.ops, "dendra_solvers", SimpleNamespace(solve_bt=chosen))
    integrator = SimpleNamespace(method="thomas", _solve=None)
    implicit._bwd_euler_bt._select_solver(integrator, "mps")
    assert integrator._solve is (chosen if native else block_pcr_solve_t)


@pytest.mark.parametrize("kind", ("scalar", "block", "multi"))
@pytest.mark.parametrize("native", (False, True))
def test_tree_mps_selection_requires_the_native_backend(monkeypatch, kind, native):
    module, implementation, operator = {
        "scalar": (tree, tree._dhs, "dhs_solve"),
        "block": (tree_bt, tree_bt._dhs_bt, "dhs_bt_solve"),
        "multi": (tree, tree._dhs_multi, "dhs_multi_solve"),
    }[kind]
    chosen = object()
    monkeypatch.setattr(module, "DENDRA_SOLVERS_AVAILABLE", True)
    monkeypatch.setattr(module, "_mps_solver_available", lambda _module: native)
    monkeypatch.setattr(
        torch.ops, "dendra_solvers", SimpleNamespace(**{operator: chosen})
    )
    integrator = SimpleNamespace(_solve=None)
    if native:
        implementation._select_solver(integrator, "mps")
        assert integrator._solve is (
            tree._dhs_multi_solve_cpu if kind == "multi" else chosen
        )
    else:
        with pytest.raises(ImportError, match="MPS-enabled"):
            implementation._select_solver(integrator, "mps")
        assert integrator._solve is None


def test_scalar_mps_spd_selection_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="currently unsupported on MPS"):
        implicit._bwd_euler_ub._select_solver(SimpleNamespace(method="spd"), "mps")


@pytest.mark.parametrize("native", (False, True))
def test_block_mps_spd_retains_portable_solver_and_functional_facade(
    monkeypatch, native
):
    monkeypatch.setattr(implicit, "DENDRA_SOLVERS_AVAILABLE", True)
    monkeypatch.setattr(implicit, "_mps_solver_available", lambda _module: native)
    integrator = SimpleNamespace(method="spd", _solve=None)
    implicit._bwd_euler_bt._select_solver(integrator, "mps")
    assert integrator._solve is block_pcr_solve_t
    assert implicit._bwd_euler_bt._functional_solver(integrator) is block_pcr_solve_t


def test_old_solver_packages_without_the_mps_probe_remain_compatible():
    assert not _mps_solver_available(None)
    assert not _mps_solver_available(SimpleNamespace())
    assert not _mps_solver_available(SimpleNamespace(is_mps_available=True))
