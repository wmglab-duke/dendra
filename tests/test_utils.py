# tests/test_utils.py

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import axonml.utils as U

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _as_1d(t: torch.Tensor) -> torch.Tensor:
    """Normalize torch outputs to 1D when they represent a single row."""
    if t.ndim == 2 and t.shape[0] == 1:
        return t[0]
    return t


def _numpy_interp_like(x, y, xnew, *, outside: str = "clamp"):
    """
    Numpy reference for 1D linear interpolation with a matching out-of-bounds policy.

    Parameters
    ----------
    outside : {"clamp", "zero"}
        "clamp" -> left=y[0], right=y[-1]
        "zero"  -> left=0,    right=0
    """
    if outside not in {"clamp", "zero"}:
        raise ValueError("outside must be 'clamp' or 'zero'")
    if outside == "clamp":
        left = y[0]
        right = y[-1]
    else:
        left = 0.0
        right = 0.0
    return np.interp(xnew, x, y, left=left, right=right)


# Strategy: sorted unique x with spacing robust under float32
float_arrays = (
    st.lists(
        st.floats(-100, 100, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
        unique=True,
    )
    .map(lambda xs: sorted(xs))
    .filter(
        lambda xs: min(
            np.float32(xs[i + 1]) - np.float32(xs[i]) for i in range(len(xs) - 1)
        )
        > 1e-6
    )
)


# --------------------------------------------------------------------------
# Functional interp1d
# --------------------------------------------------------------------------


@given(
    float_arrays,
    st.lists(
        st.floats(-100, 100, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    ),
)
@settings(max_examples=80)
def test_interp1d_matches_numpy_inside_domain(x_list, y_list):
    """Inside the domain, U.interp1d must match exact linear interpolation."""
    n = min(len(x_list), len(y_list))
    x = torch.tensor(x_list[:n], dtype=torch.float32)
    y = torch.tensor(y_list[:n], dtype=torch.float32)

    # strictly inside [min,max]
    xnew = torch.linspace(float(x.min()), float(x.max()), steps=n * 2)

    got = U.interp1d(x, y, xnew)  # outside policy irrelevant inside bounds
    got1 = _as_1d(got)

    want = torch.tensor(
        _numpy_interp_like(x.numpy(), y.numpy(), xnew.numpy(), outside="clamp"),
        dtype=torch.float32,
    )

    assert torch.allclose(got1, want, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("outside", ["zero", "clamp"])
def test_interp1d_out_of_bounds_and_grad(outside: str):
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float32, requires_grad=True)
    xq = torch.tensor([-1.0, 0.5, 3.0], dtype=torch.float32)

    yq = U.interp1d(x, y, xq, outside=outside)
    yq1 = _as_1d(yq)

    if outside == "zero":
        assert yq1[0].item() == 0.0
        assert yq1[-1].item() == 0.0
    else:
        assert torch.isclose(yq1[0], y[0])
        assert torch.isclose(yq1[-1], y[-1])

    # gradient should flow to y for the in-bounds point(s)
    y.grad = None
    yq1.sum().backward()
    assert y.grad is not None
    assert y.grad.shape == y.shape
    assert float(y.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("outside", ["zero", "clamp"])
def test_interp1d_batched_matches_numpy(outside: str):
    """
    Batched (D,N) interpolation matches NumPy row-wise under both outside policies.
    Also validates x_new broadcast when given as 1D.
    """
    torch.manual_seed(0)
    D, N, P = 4, 17, 31

    # Construct strictly increasing x per row (sorted, unique)
    base = torch.linspace(-2.0, 2.0, N, dtype=torch.float32)
    x = base.unsqueeze(0) + (0.5 * torch.arange(D, dtype=torch.float32).unsqueeze(1))
    y = torch.randn(D, N, dtype=torch.float32)

    # Row-wise query range extends outside domain
    t = torch.linspace(0.0, 1.0, P, dtype=torch.float32)
    x_min = x[:, 0:1]
    x_max = x[:, -1:]
    xnew = (x_min - 0.7) + (x_max - x_min + 1.4) * t.unsqueeze(0)  # (D,P)

    # 1) xnew is (D,P)
    got = U.interp1d(x, y, xnew, outside=outside)
    assert got.shape[0] == D and got.shape[1] == P

    want = []
    for d in range(D):
        want_d = _numpy_interp_like(
            x[d].numpy(), y[d].numpy(), xnew[d].numpy(), outside=outside
        )
        want.append(want_d)
    want = torch.tensor(np.stack(want, axis=0), dtype=torch.float32)

    assert torch.allclose(got, want, atol=2e-5, rtol=1e-4)

    # 2) xnew is 1D broadcast to all rows (functional version allows this)
    xnew_1d = xnew[0].clone()  # (P,)
    got_b = U.interp1d(x, y, xnew_1d, outside=outside)
    assert got_b.shape[0] == D and got_b.shape[1] == P

    want_b = []
    for d in range(D):
        want_d = _numpy_interp_like(
            x[d].numpy(), y[d].numpy(), xnew_1d.numpy(), outside=outside
        )
        want_b.append(want_d)
    want_b = torch.tensor(np.stack(want_b, axis=0), dtype=torch.float32)

    assert torch.allclose(got_b, want_b, atol=2e-5, rtol=1e-4)


# --------------------------------------------------------------------------
# PreparedInterp1d
# --------------------------------------------------------------------------


@pytest.mark.parametrize("outside", ["zero", "clamp"])
def test_prepared_unbatched_matches_numpy(outside: str):
    torch.manual_seed(1)
    N, P = 19, 37

    x = torch.sort(torch.randn(N, dtype=torch.float32)).values
    y = torch.randn(N, dtype=torch.float32)

    xq = torch.linspace(
        float(x.min()) - 0.5, float(x.max()) + 0.5, P, dtype=torch.float32
    )
    xq2 = torch.stack([xq, xq + 0.1], dim=0)  # (2,P)

    interp = U.PreparedInterp1d(x, y, outside=outside, sort_xy=True)

    got1 = interp(xq)
    got2 = interp(xq2)

    got1_ = _as_1d(got1).detach().cpu().numpy()
    got2_ = got2.detach().cpu().numpy()

    want1 = _numpy_interp_like(x.numpy(), y.numpy(), xq.numpy(), outside=outside)
    want2 = np.stack(
        [
            _numpy_interp_like(x.numpy(), y.numpy(), xq.numpy(), outside=outside),
            _numpy_interp_like(
                x.numpy(), y.numpy(), (xq.numpy() + 0.1), outside=outside
            ),
        ],
        axis=0,
    )

    assert np.allclose(got1_, want1, atol=2e-5, rtol=1e-4)
    assert np.allclose(got2_, want2, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("outside", ["zero", "clamp"])
def test_prepared_batched_aligned_matches_numpy(outside: str):
    torch.manual_seed(2)
    D, N, P = 3, 23, 41

    # Strictly increasing x per row via cumulative sum
    steps = torch.rand(D, N, dtype=torch.float32) + 0.1
    x = torch.cumsum(steps, dim=1)
    y = torch.randn(D, N, dtype=torch.float32)

    # Aligned x_new shape (D,P), includes out-of-bounds
    t = torch.linspace(0.0, 1.0, P, dtype=torch.float32)
    x_min = x[:, :1]
    x_max = x[:, -1:]
    xnew = (x_min - 0.25) + (x_max - x_min + 0.5) * t.unsqueeze(0)

    interp = U.PreparedInterp1d(x, y, outside=outside, sort_xy=True)
    got = interp(xnew)  # indices None => aligned
    assert got.shape == (D, P)

    want = []
    for d in range(D):
        want.append(
            _numpy_interp_like(
                x[d].numpy(), y[d].numpy(), xnew[d].numpy(), outside=outside
            )
        )
    want = torch.tensor(np.stack(want, axis=0), dtype=torch.float32)

    assert torch.allclose(got, want, atol=2e-5, rtol=1e-4)


def test_prepared_requires_indices_when_ambiguous():
    torch.manual_seed(3)
    D, N, P = 4, 11, 13
    x = torch.cumsum(torch.rand(D, N, dtype=torch.float32) + 0.1, dim=1)
    y = torch.randn(D, N, dtype=torch.float32)

    interp = U.PreparedInterp1d(x, y, outside="zero", sort_xy=True)

    # x_new 1D with batched LUTs is ambiguous without indices
    xq_1d = torch.linspace(
        float(x.min()) - 0.1, float(x.max()) + 0.1, P, dtype=torch.float32
    )
    with pytest.raises(ValueError):
        _ = interp(xq_1d)

    # x_new 2D with Q != D is ambiguous without indices
    xq_2d = torch.randn(D + 1, P, dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = interp(xq_2d)


@pytest.mark.parametrize("outside", ["zero", "clamp"])
def test_prepared_indices_mapping_matches_numpy(outside: str):
    torch.manual_seed(4)
    D, N, P = 5, 17, 29

    x = torch.cumsum(torch.rand(D, N, dtype=torch.float32) + 0.1, dim=1)
    y = torch.randn(D, N, dtype=torch.float32)
    interp = U.PreparedInterp1d(x, y, outside=outside, sort_xy=True)

    # Case A: x_new is (Q,P), map each row via indices
    Q = 7
    idx = torch.randint(0, D, (Q,), dtype=torch.long)
    xq = torch.randn(Q, P, dtype=torch.float32)  # not necessarily within range

    got = interp(xq, indices=idx)
    assert got.shape == (Q, P)

    want = []
    for q in range(Q):
        d = int(idx[q])
        want.append(
            _numpy_interp_like(
                x[d].numpy(), y[d].numpy(), xq[q].numpy(), outside=outside
            )
        )
    want = torch.tensor(np.stack(want, axis=0), dtype=torch.float32)

    assert torch.allclose(got, want, atol=2e-5, rtol=1e-4)

    # Case B: x_new is 1D (P,), indices selects L rows => output is (L,P) if L>1
    xq1 = torch.linspace(
        float(x.min()) - 0.2, float(x.max()) + 0.2, P, dtype=torch.float32
    )
    idx2 = torch.tensor([0, 3, 3, 4], dtype=torch.long)  # L=4
    got2 = interp(xq1, indices=idx2)
    assert got2.shape == (len(idx2), P)

    want2 = []
    for d in idx2.tolist():
        want2.append(
            _numpy_interp_like(x[d].numpy(), y[d].numpy(), xq1.numpy(), outside=outside)
        )
    want2 = torch.tensor(np.stack(want2, axis=0), dtype=torch.float32)

    assert torch.allclose(got2, want2, atol=2e-5, rtol=1e-4)


def test_prepared_sort_xy_corrects_unsorted_input():
    """
    PreparedInterp1d(sort_xy=True) should work even if x is unsorted by sorting x and permuting y.
    """
    # 1D example
    x = torch.tensor([2.0, 0.0, 1.0], dtype=torch.float32)
    y = torch.tensor([20.0, 0.0, 10.0], dtype=torch.float32)
    xq = torch.tensor([-1.0, 0.5, 3.0], dtype=torch.float32)

    interp = U.PreparedInterp1d(x, y, outside="clamp", sort_xy=True)
    got = _as_1d(interp(xq)).detach().cpu().numpy()

    xs, perm = torch.sort(x)
    ys = y.index_select(0, perm)
    want = _numpy_interp_like(xs.numpy(), ys.numpy(), xq.numpy(), outside="clamp")

    assert np.allclose(got, want, atol=2e-5, rtol=1e-4)

    # 2D example
    x2 = torch.tensor([[2.0, 0.0, 1.0], [3.0, 1.0, 2.0]], dtype=torch.float32)
    y2 = torch.tensor([[20.0, 0.0, 10.0], [30.0, 10.0, 20.0]], dtype=torch.float32)
    xq2 = torch.tensor([[0.5, 2.5], [1.5, 3.5]], dtype=torch.float32)

    interp2 = U.PreparedInterp1d(x2, y2, outside="clamp", sort_xy=True)
    got2 = interp2(xq2).detach().cpu().numpy()

    want2 = []
    for d in range(x2.shape[0]):
        xs, perm = torch.sort(x2[d])
        ys = y2[d].index_select(0, perm)
        want2.append(
            _numpy_interp_like(xs.numpy(), ys.numpy(), xq2[d].numpy(), outside="clamp")
        )
    want2 = np.stack(want2, axis=0)

    assert np.allclose(got2, want2, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("outside", ["zero", "clamp"])
def test_prepared_learnable_y_grad(outside: str):
    torch.manual_seed(5)
    N, P = 13, 23

    x = torch.sort(torch.randn(N, dtype=torch.float32)).values
    y = torch.randn(N, dtype=torch.float32)

    interp = U.PreparedInterp1d(x, y, outside=outside, sort_xy=True, learnable_y=True)

    xq = torch.linspace(
        float(x.min()) - 0.3, float(x.max()) + 0.3, P, dtype=torch.float32
    )
    out = interp(xq)
    loss = _as_1d(out).pow(2).mean()

    loss.backward()

    assert hasattr(interp, "y")
    assert interp.y.grad is not None
    assert interp.y.grad.shape == interp.y.shape
    assert float(interp.y.grad.abs().sum()) > 0.0
