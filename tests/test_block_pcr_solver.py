"""Portable block-PCR correctness and ExtCell MPS dispatch contracts."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
import dendra.models.integrators.implicit as implicit
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.implicit import _bwd_euler_bt
from dendra.models.integrators.tridiag.block import block_pcr_solve_t
from dendra.models.mod import pas

MPS_AVAILABLE = bool(
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
)


def _dense_block_matrix(lower, main, upper):
    batch_shape = tuple(main.shape[:-3])
    system_size, block_size = main.shape[-3:-1]
    dense = torch.zeros(
        batch_shape + (system_size * block_size, system_size * block_size),
        device=main.device,
        dtype=main.dtype,
    )
    lower = torch.diag_embed(lower) if lower.ndim == main.ndim - 1 else lower
    upper = torch.diag_embed(upper) if upper.ndim == main.ndim - 1 else upper
    for index in range(system_size):
        start = index * block_size
        stop = start + block_size
        dense[..., start:stop, start:stop] = main[..., index, :, :]
    for index in range(system_size - 1):
        start = index * block_size
        middle = start + block_size
        stop = middle + block_size
        dense[..., middle:stop, start:middle] = lower[..., index, :, :]
        dense[..., start:middle, middle:stop] = upper[..., index, :, :]
    return dense


def _well_conditioned_case(system_size, *, full_bands=False, device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(7300 + system_size)
    batch_size, block_size = 2, 3
    main = 0.04 * torch.randn(
        batch_size,
        system_size,
        block_size,
        block_size,
        generator=generator,
        dtype=torch.float64,
    )
    main = main + 4.0 * torch.eye(block_size, dtype=torch.float64)
    band_shape = (batch_size, system_size - 1, block_size)
    if full_bands:
        band_shape += (block_size,)
    lower = 0.03 * torch.randn(band_shape, generator=generator, dtype=torch.float64)
    upper = 0.03 * torch.randn(band_shape, generator=generator, dtype=torch.float64)
    rhs = torch.randn(
        batch_size,
        system_size,
        block_size,
        generator=generator,
        dtype=torch.float64,
    )
    return tuple(
        value.to(device=device, dtype=torch.float32 if device == "mps" else None)
        for value in (lower, main, upper, rhs)
    )


@pytest.mark.parametrize("system_size", [1, 2, 3, 5, 8, 13])
@pytest.mark.parametrize("full_bands", [False, True])
def test_block_pcr_matches_dense_for_arbitrary_lengths_and_band_forms(
    system_size, full_bands
):
    lower, main, upper, rhs = _well_conditioned_case(system_size, full_bands=full_bands)
    originals = tuple(value.clone() for value in (lower, main, upper, rhs))

    actual = block_pcr_solve_t(lower, main, upper, rhs)
    dense = _dense_block_matrix(lower, main, upper)
    expected = torch.linalg.solve(dense, rhs.reshape(2, system_size * 3, 1)).reshape_as(
        rhs
    )

    torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-13)
    for value, original in zip((lower, main, upper, rhs), originals):
        assert torch.equal(value, original)


def test_block_pcr_is_differentiable_through_all_inputs():
    lower, main, upper, rhs = _well_conditioned_case(5)
    inputs = tuple(value.requires_grad_() for value in (lower, main, upper, rhs))
    assert torch.autograd.gradcheck(
        block_pcr_solve_t,
        inputs,
        eps=1.0e-6,
        atol=1.0e-5,
        rtol=1.0e-4,
    )


def test_block_pcr_is_capturable_as_one_full_torch_graph():
    inputs = _well_conditioned_case(13)
    expected = block_pcr_solve_t(*inputs)
    compiled = torch.compile(block_pcr_solve_t, backend="eager", fullgraph=True)

    actual = compiled(*inputs)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class _SelectorModel(torch.nn.Module):
    """Minimum model surface needed to construct the block integrator."""

    def __init__(self):
        super().__init__()
        self.shape = (1, 3)
        self.n_layers = 2
        self.v_init = -65.0
        self.reported_device = torch.device("cpu")
        self.register_buffer("v", torch.full(self.shape, self.v_init))

    def device(self):
        return self.reported_device

    def dtype(self):
        return self.v.dtype


@pytest.mark.parametrize("method", ["thomas", "spd"])
def test_block_integrator_selects_portable_solver_for_mps_without_allocating_mps(
    monkeypatch, method
):
    monkeypatch.setattr(implicit, "DENDRA_SOLVERS_AVAILABLE", False)
    model = _SelectorModel()
    integrator = _bwd_euler_bt(model, torch.nn.Module(), method=method)
    model.reported_device = torch.device("mps")

    integrator._select_solver(model.device())

    assert integrator._solve is block_pcr_solve_t


@pytest.mark.parametrize("method", ["thomas", "spd"])
def test_block_integrator_preserves_cpu_extension_requirement(monkeypatch, method):
    monkeypatch.setattr(implicit, "DENDRA_SOLVERS_AVAILABLE", False)
    model = _SelectorModel()
    integrator = _bwd_euler_bt(model, torch.nn.Module(), method=method)

    with pytest.raises(ImportError, match="requires dendra_solvers for CPU"):
        integrator._select_solver(model.device())


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS is not available")
def test_block_pcr_mps_matches_cpu_dense_reference():
    system_size = 13
    lower, main, upper, rhs = _well_conditioned_case(system_size, device="mps")

    actual = block_pcr_solve_t(lower, main, upper, rhs)
    dense = _dense_block_matrix(lower.cpu(), main.cpu(), upper.cpu())
    expected = torch.linalg.solve(
        dense, rhs.cpu().reshape(2, system_size * 3, 1)
    ).reshape_as(rhs.cpu())

    assert actual.device.type == "mps"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, rtol=1.0e-4, atol=2.0e-5)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS is not available")
def test_block_pcr_mps_is_inductor_fullgraph_compatible():
    from torch._inductor import config as inductor_config

    inputs = _well_conditioned_case(13, device="mps")
    expected = block_pcr_solve_t(*inputs)
    # PyTorch's non-AOT MPS wrapper currently requires cpp_wrapper=False.
    with (
        torch_compiler_warning_context(),
        inductor_config.patch({"cpp_wrapper": False}),
    ):
        compiled = torch.compile(
            block_pcr_solve_t,
            backend="inductor",
            fullgraph=True,
        )
        actual = compiled(*inputs)

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=2.0e-6)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS is not available")
def test_extcell_axon_initializes_and_steps_with_mps_block_solver():
    axon = dn.ExtCellAxon(
        diameters=[6.0],
        n_comp=5,
        device="mps",
        dtype=torch.float32,
    )
    axon.xraxial.fill_(2.0)
    axon.xc.fill_(0.25)
    axon.xg.fill_(1.0e-4)
    axon.insert(pas, g=1.0e-3, e=-70.0)

    axon.initialize()
    axon.run(tstop=0.01, dt=0.005)

    assert axon.integrator._solve is block_pcr_solve_t
    assert axon.v.device.type == "mps"
    assert torch.isfinite(axon.v).all()
    torch.testing.assert_close(axon.v, axon.vc[..., 0] - axon.vc[..., 1])
