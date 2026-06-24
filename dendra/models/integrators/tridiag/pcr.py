import torch


def pcr_solve_t(
    a: torch.Tensor,  # (..., K-1) subdiag
    b: torch.Tensor,  # (..., K)   diag
    c: torch.Tensor,  # (..., K-1) superdiag
    d: torch.Tensor,  # (..., K)   RHS
    switch_to_thomas_at: int | None = None,
) -> torch.Tensor:
    """Solve a batched tridiagonal system on CPU/PyTorch.

    This function is used as Dendra's pure-PyTorch fallback for tridiagonal
    cable solves.  The previous partial-PCR reduction was only exact for some
    system sizes and produced visible errors for prime/non-power-of-two lengths.
    The fallback now uses a functional batched Thomas algorithm, preserving the
    public ``pcr_solve_t`` name while prioritizing numerical correctness and
    autograd-safety on CPU.

    Parameters
    ----------
    a, b, c, d
        Tridiagonal bands and RHS with matching leading batch dimensions.
        ``a`` and ``c`` have trailing length ``K-1``; ``b`` and ``d`` have
        trailing length ``K``.
    switch_to_thomas_at
        Accepted for API compatibility; ignored by this robust fallback.

    Returns
    -------
    torch.Tensor
        Solution tensor with the same shape as ``d``.
    """
    del switch_to_thomas_at

    if b.shape != d.shape:
        raise ValueError(
            f"b and d must have the same shape; got {b.shape} and {d.shape}."
        )
    if b.ndim < 1:
        raise ValueError("b/d must have at least one dimension.")

    K = int(b.shape[-1])
    if K == 0:
        raise ValueError("tridiagonal systems must have K >= 1.")
    if K == 1:
        return d / b

    expected_offdiag = b.shape[:-1] + (K - 1,)
    if a.shape != expected_offdiag or c.shape != expected_offdiag:
        raise ValueError(
            "a and c must have shape b.shape[:-1] + (K - 1,); "
            f"got a={a.shape}, c={c.shape}, b={b.shape}."
        )

    # Functional Thomas sweep.  Avoid in-place writes so autograd does not have
    # to track versioned mutated intermediates through long BPTT rollouts.
    cp = []
    dp = []

    denom0 = b[..., 0]
    cp.append(c[..., 0] / denom0)
    dp.append(d[..., 0] / denom0)

    for i in range(1, K):
        denom = b[..., i] - a[..., i - 1] * cp[i - 1]
        if i < K - 1:
            cp.append(c[..., i] / denom)
        dp.append((d[..., i] - a[..., i - 1] * dp[i - 1]) / denom)

    x = [None] * K
    x[-1] = dp[-1]
    for i in range(K - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return torch.stack(x, dim=-1)


def _batched_thomas_inplace(a_full, b_full, c_full, d_full):
    """Legacy helper retained for backward compatibility with private callers.

    ``a_full`` and ``c_full`` are full-length bands with zero boundary entries.
    The solution is written into ``d_full``.
    """
    B, m = b_full.shape
    if m == 0:
        return
    c_full[:, 0] = c_full[:, 0] / b_full[:, 0]
    d_full[:, 0] = d_full[:, 0] / b_full[:, 0]
    for i in range(1, m):
        denom = b_full[:, i] - a_full[:, i] * c_full[:, i - 1]
        if i < m - 1:
            c_full[:, i] = c_full[:, i] / denom
        d_full[:, i] = (d_full[:, i] - a_full[:, i] * d_full[:, i - 1]) / denom
    for i in range(m - 2, -1, -1):
        d_full[:, i] = d_full[:, i] - c_full[:, i] * d_full[:, i + 1]
