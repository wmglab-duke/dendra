import torch


def _solve_linear_small(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve A x = b for small dense systems, batched.

    A: (..., n, n)
    b: (..., n)
    returns x: (..., n)

    Uses explicit elimination for n=1..4, falls back to torch.linalg.solve otherwise.
    """
    n = A.shape[-1]
    if n == 1:
        return b / A[..., 0, 0]

    if n == 2:
        a00 = A[..., 0, 0]
        a01 = A[..., 0, 1]
        a10 = A[..., 1, 0]
        a11 = A[..., 1, 1]
        b0 = b[..., 0]
        b1 = b[..., 1]
        det = a00 * a11 - a01 * a10
        x0 = (a11 * b0 - a01 * b1) / det
        x1 = (-a10 * b0 + a00 * b1) / det
        return torch.stack((x0, x1), dim=-1)

    if n == 3:
        a00 = A[..., 0, 0]
        a01 = A[..., 0, 1]
        a02 = A[..., 0, 2]
        a10 = A[..., 1, 0]
        a11 = A[..., 1, 1]
        a12 = A[..., 1, 2]
        a20 = A[..., 2, 0]
        a21 = A[..., 2, 1]
        a22 = A[..., 2, 2]
        b0 = b[..., 0]
        b1 = b[..., 1]
        b2 = b[..., 2]

        m10 = a10 / a00
        m20 = a20 / a00

        a11_ = a11 - m10 * a01
        a12_ = a12 - m10 * a02
        b1_ = b1 - m10 * b0

        a21_ = a21 - m20 * a01
        a22_ = a22 - m20 * a02
        b2_ = b2 - m20 * b0

        m21 = a21_ / a11_
        a22__ = a22_ - m21 * a12_
        b2__ = b2_ - m21 * b1_

        x2 = b2__ / a22__
        x1 = (b1_ - a12_ * x2) / a11_
        x0 = (b0 - a01 * x1 - a02 * x2) / a00
        return torch.stack((x0, x1, x2), dim=-1)

    if n == 4:
        a00 = A[..., 0, 0]
        a01 = A[..., 0, 1]
        a02 = A[..., 0, 2]
        a03 = A[..., 0, 3]
        a10 = A[..., 1, 0]
        a11 = A[..., 1, 1]
        a12 = A[..., 1, 2]
        a13 = A[..., 1, 3]
        a20 = A[..., 2, 0]
        a21 = A[..., 2, 1]
        a22 = A[..., 2, 2]
        a23 = A[..., 2, 3]
        a30 = A[..., 3, 0]
        a31 = A[..., 3, 1]
        a32 = A[..., 3, 2]
        a33 = A[..., 3, 3]
        b0 = b[..., 0]
        b1 = b[..., 1]
        b2 = b[..., 2]
        b3 = b[..., 3]

        m10 = a10 / a00
        a11_ = a11 - m10 * a01
        a12_ = a12 - m10 * a02
        a13_ = a13 - m10 * a03
        b1_ = b1 - m10 * b0

        m20 = a20 / a00
        a21_ = a21 - m20 * a01
        a22_ = a22 - m20 * a02
        a23_ = a23 - m20 * a03
        b2_ = b2 - m20 * b0

        m30 = a30 / a00
        a31_ = a31 - m30 * a01
        a32_ = a32 - m30 * a02
        a33_ = a33 - m30 * a03
        b3_ = b3 - m30 * b0

        m21 = a21_ / a11_
        a22__ = a22_ - m21 * a12_
        a23__ = a23_ - m21 * a13_
        b2__ = b2_ - m21 * b1_

        m31 = a31_ / a11_
        a32__ = a32_ - m31 * a12_
        a33__ = a33_ - m31 * a13_
        b3__ = b3_ - m31 * b1_

        m32 = a32__ / a22__
        a33___ = a33__ - m32 * a23__
        b3___ = b3__ - m32 * b2__

        x3 = b3___ / a33___
        x2 = (b2__ - a23__ * x3) / a22__
        x1 = (b1_ - a12_ * x2 - a13_ * x3) / a11_
        x0 = (b0 - a01 * x1 - a02 * x2 - a03 * x3) / a00
        return torch.stack((x0, x1, x2, x3), dim=-1)

    # Fallback for larger systems
    return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
