"""Pure-PyTorch solvers for block-tridiagonal systems."""

from __future__ import annotations

import torch


def _as_block_band(
    band: torch.Tensor,
    *,
    name: str,
    batch_shape: tuple[int, ...],
    system_size: int,
    block_size: int,
) -> torch.Tensor:
    """Normalize diagonal-vector or full-matrix off-diagonal blocks."""
    vector_shape = batch_shape + (system_size - 1, block_size)
    matrix_shape = vector_shape + (block_size,)
    if tuple(band.shape) == vector_shape:
        return torch.diag_embed(band)
    if tuple(band.shape) == matrix_shape:
        return band
    raise ValueError(
        f"{name} must have shape {vector_shape} for diagonal blocks or "
        f"{matrix_shape} for full blocks; got {tuple(band.shape)}."
    )


def _inverse_3x3(matrix: torch.Tensor) -> torch.Tensor:
    """Differentiable, compile-portable inverse for ExtCell's 3x3 blocks."""
    # Normalization keeps cofactors and the determinant in a safe float32 range
    # for the differently scaled membrane/extracellular circuit coefficients.
    scale = matrix.abs().amax(dim=(-2, -1), keepdim=True)
    value = matrix / scale
    a, b, c = value.unbind(dim=-2)
    a0, a1, a2 = a.unbind(dim=-1)
    b0, b1, b2 = b.unbind(dim=-1)
    c0, c1, c2 = c.unbind(dim=-1)

    adjugate = torch.stack(
        (
            b1 * c2 - b2 * c1,
            a2 * c1 - a1 * c2,
            a1 * b2 - a2 * b1,
            b2 * c0 - b0 * c2,
            a0 * c2 - a2 * c0,
            a2 * b0 - a0 * b2,
            b0 * c1 - b1 * c0,
            a1 * c0 - a0 * c1,
            a0 * b1 - a1 * b0,
        ),
        dim=-1,
    ).reshape(matrix.shape)
    determinant = a0 * adjugate[..., 0, 0]
    determinant = determinant + a1 * adjugate[..., 1, 0]
    determinant = determinant + a2 * adjugate[..., 2, 0]
    return adjugate / determinant[..., None, None] / scale


def _solve_blocks(coefficient: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve small block systems without MPS's compiled solve-layout bug."""
    if coefficient.shape[-1] == 3:
        inverse = _inverse_3x3(coefficient)
        if rhs.ndim == coefficient.ndim - 1:
            return (inverse @ rhs.unsqueeze(-1)).squeeze(-1)
        return inverse @ rhs
    return torch.linalg.solve(coefficient, rhs)


def _right_solve(coefficient: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Return ``factor @ inv(coefficient)``."""
    if coefficient.shape[-1] == 3:
        return factor @ _inverse_3x3(coefficient)
    return _solve_blocks(
        coefficient.transpose(-1, -2).contiguous(),
        factor.transpose(-1, -2).contiguous(),
    ).transpose(-1, -2)


def block_pcr_solve_t(
    lower: torch.Tensor,
    main: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Solve a batch of block-tridiagonal systems using block PCR.

    This is Dendra's device-portable fallback for accelerators without a native
    block solver, notably Apple MPS. Parallel cyclic reduction eliminates both
    neighbours of every block row at once, so a system with ``K`` rows needs
    only ``ceil(log2(K))`` tensor-level reduction rounds. This is substantially
    friendlier to accelerator execution and ``torch.compile`` than unrolling a
    sequential block-Thomas sweep over every compartment.

    Parameters
    ----------
    lower, upper
        Off-diagonal blocks. Each may be supplied either as diagonal vectors
        with shape ``(*batch, K - 1, M)`` or as general blocks with shape
        ``(*batch, K - 1, M, M)``.
    main
        Main blocks with shape ``(*batch, K, M, M)``.
    rhs
        Right-hand side with shape ``(*batch, K, M)``.

    Returns
    -------
    torch.Tensor
        The solution, with the same shape, dtype, and device as ``rhs``.

    Notes
    -----
    The implementation is functional: it does not mutate any input and remains
    differentiable through its batched block solves. ``K`` need not be a power
    of two.
    """
    tensors = {"lower": lower, "main": main, "upper": upper, "rhs": rhs}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")

    if main.ndim < 3:
        raise ValueError(
            f"main must have shape (*batch, K, M, M); got {tuple(main.shape)}."
        )
    if main.shape[-1] != main.shape[-2]:
        raise ValueError(f"main blocks must be square; got {tuple(main.shape)}.")

    batch_shape = tuple(main.shape[:-3])
    system_size = int(main.shape[-3])
    block_size = int(main.shape[-1])
    if system_size < 1:
        raise ValueError("block-tridiagonal systems must have K >= 1.")
    if block_size < 1:
        raise ValueError("block-tridiagonal systems must have M >= 1.")

    expected_rhs = batch_shape + (system_size, block_size)
    if tuple(rhs.shape) != expected_rhs:
        raise ValueError(f"rhs must have shape {expected_rhs}; got {tuple(rhs.shape)}.")

    for name, tensor in tensors.items():
        if tensor.device != main.device:
            raise ValueError(
                "all block-tridiagonal inputs must be on the same device; "
                f"main is on {main.device}, but {name} is on {tensor.device}."
            )
        if tensor.dtype != main.dtype:
            raise TypeError(
                "all block-tridiagonal inputs must have the same dtype; "
                f"main is {main.dtype}, but {name} is {tensor.dtype}."
            )
    if not (torch.is_floating_point(main) or torch.is_complex(main)):
        raise TypeError(
            "block-tridiagonal inputs must have a floating-point or complex dtype."
        )

    lower_blocks = _as_block_band(
        lower,
        name="lower",
        batch_shape=batch_shape,
        system_size=system_size,
        block_size=block_size,
    )
    upper_blocks = _as_block_band(
        upper,
        name="upper",
        batch_shape=batch_shape,
        system_size=system_size,
        block_size=block_size,
    )

    if system_size == 1:
        return _solve_blocks(main, rhs)

    zero_boundary = torch.zeros_like(main[..., :1, :, :])
    a = torch.cat((zero_boundary, lower_blocks), dim=-3)
    b = main
    c = torch.cat((upper_blocks, zero_boundary), dim=-3)
    d = rhs

    identity = torch.eye(block_size, dtype=main.dtype, device=main.device)
    identity = identity.reshape((1,) * len(batch_shape) + (1, block_size, block_size))

    stride = 1
    while stride < system_size:
        zero_blocks = torch.zeros_like(b[..., :stride, :, :])
        zero_vectors = torch.zeros_like(d[..., :stride, :])
        identity_blocks = identity.expand(
            batch_shape + (stride, block_size, block_size)
        )

        # For row i, these tensors contain row i-stride (left) or i+stride
        # (right). Identity padding keeps the auxiliary boundary solves valid;
        # the corresponding a/c block is zero and therefore cancels the result.
        b_left = torch.cat((identity_blocks, b[..., :-stride, :, :]), dim=-3)
        b_right = torch.cat((b[..., stride:, :, :], identity_blocks), dim=-3)

        a_left = torch.cat((zero_blocks, a[..., :-stride, :, :]), dim=-3)
        c_left = torch.cat((zero_blocks, c[..., :-stride, :, :]), dim=-3)
        d_left = torch.cat((zero_vectors, d[..., :-stride, :]), dim=-2)

        a_right = torch.cat((a[..., stride:, :, :], zero_blocks), dim=-3)
        c_right = torch.cat((c[..., stride:, :, :], zero_blocks), dim=-3)
        d_right = torch.cat((d[..., stride:, :], zero_vectors), dim=-2)

        alpha = -_right_solve(b_left, a)
        beta = -_right_solve(b_right, c)

        b = b + alpha @ c_left + beta @ a_right
        d = (
            d
            + (alpha @ d_left.unsqueeze(-1)).squeeze(-1)
            + (beta @ d_right.unsqueeze(-1)).squeeze(-1)
        )
        a = alpha @ a_left
        c = beta @ c_right
        stride *= 2

    return _solve_blocks(b, d)
