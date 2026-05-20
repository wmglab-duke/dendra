import torch


def pcr_solve_t(
    a: torch.Tensor,  # (B, K-1) subdiag
    b: torch.Tensor,  # (B, K)   diag
    c: torch.Tensor,  # (B, K-1) superdiag
    d: torch.Tensor,  # (B, K)   RHS
    switch_to_thomas_at: int | None = None,  # e.g., 64 or 32
) -> torch.Tensor:
    B, K = b.shape
    dev, dtype = b.device, b.dtype

    a_full = torch.zeros(B, K, device=dev, dtype=dtype)
    c_full = torch.zeros(B, K, device=dev, dtype=dtype)
    a_full[:, 1:] = a
    c_full[:, :-1] = c

    b_full = b.clone()
    d_full = d.clone()

    stride = 1
    # only valid while K - 2*stride > 0  =>  2*stride < K
    while 2 * stride < K and (
        switch_to_thomas_at is None or stride < switch_to_thomas_at
    ):
        iL = slice(stride, K - stride)
        iLL = slice(0, K - 2 * stride)
        iRR = slice(2 * stride, K)

        alpha = a_full[:, iL] / b_full[:, iLL]
        gamma = c_full[:, iL] / b_full[:, iRR]

        aL = a_full[:, iLL]
        cL = c_full[:, iLL]
        dL = d_full[:, iLL]

        aR = a_full[:, iRR]
        cR = c_full[:, iRR]
        dR = d_full[:, iRR]

        # central row updates
        b_full[:, iL].addcmul_(cL, -alpha).addcmul_(aR, -gamma)
        d_full[:, iL].addcmul_(dL, -alpha).addcmul_(dR, -gamma)
        a_full[:, iL].copy_(-aL * alpha)
        c_full[:, iL].copy_(-cR * gamma)

        # keep edges tri-diagonal, they lose one neighbor
        if stride > 0:
            a_full[:, :stride].zero_()
            c_full[:, -stride:].zero_()

        stride <<= 1

    # Always finish with Thomas on segments of length <= 2*stride
    seg = min(2 * stride, K)
    for s in range(0, K, seg):
        e = min(s + seg, K)
        bb = b_full[:, s:e]
        dd = d_full[:, s:e]
        aa = a_full[:, s:e]
        cc = c_full[:, s:e]
        _batched_thomas_inplace(aa, bb, cc, dd)

    # Thomas wrote x in d_full
    return d_full


def _batched_thomas_inplace(a_full, b_full, c_full, d_full):
    """
    In-place Thomas on blocks: a_full has size (B, m) with a_full[:,0]==0,
    c_full[:,m-1]==0. Writes solution into d_full; keeps b_full as diag.
    """
    B, m = b_full.shape
    # forward sweep
    c_full[:, 0] = c_full[:, 0] / b_full[:, 0]
    d_full[:, 0] = d_full[:, 0] / b_full[:, 0]
    for i in range(1, m):
        denom = b_full[:, i] - a_full[:, i] * c_full[:, i - 1]
        if i < m - 1:
            c_full[:, i] = c_full[:, i] / denom
        d_full[:, i] = (d_full[:, i] - a_full[:, i] * d_full[:, i - 1]) / denom
    # back substitution
    for i in range(m - 2, -1, -1):
        d_full[:, i] = d_full[:, i] - c_full[:, i] * d_full[:, i + 1]
