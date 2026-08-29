# tests/test_utils.py

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import dendra.utils as U
from dendra.utils.tensor_ops import _logical_tensor_bytes

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_logical_tensor_bytes_materializes_singleton_expanded_layout(dtype):
    value = torch.tensor(2.0, dtype=dtype).expand(1, 1)
    assert value.stride() == (0, 0)

    actual = _logical_tensor_bytes(value)
    expected = torch.tensor(2.0, dtype=dtype).reshape(-1).view(torch.uint8)

    assert actual.device.type == "cpu"
    assert actual.dtype == torch.uint8
    assert actual.stride() == (1,)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def _as_1d(t: torch.Tensor) -> torch.Tensor:
    """Normalize torch outputs to 1D when they represent a single row."""
    if t.ndim == 2 and t.shape[0] == 1:
        return t[0]
    return t


def _numpy_interp_like(x, y, xnew, *, outside: str = "clamp", fill_value: float = 0.0):
    """
    Numpy reference for 1D linear interpolation with a matching out-of-bounds policy.

    Parameters
    ----------
    outside : {"clamp", "zero", "fill"}
        "clamp" -> left=y[0],      right=y[-1]
        "zero"  -> left=0,         right=0
        "fill"  -> left=fill_value right=fill_value
    """
    if outside not in {"clamp", "zero", "fill"}:
        raise ValueError("outside must be 'clamp', 'zero', or 'fill'")
    if outside == "clamp":
        left = y[0]
        right = y[-1]
    elif outside == "zero":
        left = 0.0
        right = 0.0
    else:
        left = float(fill_value)
        right = float(fill_value)
    return np.interp(xnew, x, y, left=left, right=right)


def _numpy_trilinear_rect(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    xyz_new: np.ndarray,
    *,
    outside: str = "clamp",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Reference trilinear interpolation on a rectilinear grid.

    - x,y,z are 1D strictly increasing
    - values shape: (Nx,Ny,Nz) or (Nx,Ny,Nz,C)
    - xyz_new shape: (P,3)
    - outside: "clamp" => border semantics by clamping coords
               "zero"  => any coord OOB => 0
               "fill"  => any coord OOB => fill_value
    """
    if outside not in {"clamp", "zero", "fill"}:
        raise ValueError("outside must be 'clamp', 'zero', or 'fill'")

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    xyz_new = np.asarray(xyz_new, dtype=np.float32)

    vals = np.asarray(values, dtype=np.float32)
    had_channels = vals.ndim == 4
    if not had_channels:
        vals = vals[..., None]  # (Nx,Ny,Nz,1)

    Nx, Ny, Nz, C = vals.shape
    P = xyz_new.shape[0]
    out = np.empty((P, C), dtype=np.float32)

    x_min, x_max = x[0], x[-1]
    y_min, y_max = y[0], y[-1]
    z_min, z_max = z[0], z[-1]

    def _ind_t(axis: np.ndarray, q: float) -> tuple[int, float]:
        # assumes q is in-bounds (or already clamped)
        i = int(np.searchsorted(axis, q) - 1)
        if i < 0:
            i = 0
        hi = axis.shape[0] - 2
        if i > hi:
            i = hi
        denom = float(axis[i + 1] - axis[i])
        if denom == 0.0:
            t = 0.0
        else:
            t = float((q - axis[i]) / denom)
        return i, t

    for p in range(P):
        qx, qy, qz = float(xyz_new[p, 0]), float(xyz_new[p, 1]), float(xyz_new[p, 2])

        if outside == "clamp":
            qx = float(np.clip(qx, x_min, x_max))
            qy = float(np.clip(qy, y_min, y_max))
            qz = float(np.clip(qz, z_min, z_max))
        else:
            inside = (
                (x_min <= qx <= x_max)
                and (y_min <= qy <= y_max)
                and (z_min <= qz <= z_max)
            )
            if not inside:
                out[p, :] = 0.0 if outside == "zero" else float(fill_value)
                continue

        ix, tx = _ind_t(x, qx)
        iy, ty = _ind_t(y, qy)
        iz, tz = _ind_t(z, qz)

        v000 = vals[ix, iy, iz, :]
        v100 = vals[ix + 1, iy, iz, :]
        v010 = vals[ix, iy + 1, iz, :]
        v110 = vals[ix + 1, iy + 1, iz, :]
        v001 = vals[ix, iy, iz + 1, :]
        v101 = vals[ix + 1, iy, iz + 1, :]
        v011 = vals[ix, iy + 1, iz + 1, :]
        v111 = vals[ix + 1, iy + 1, iz + 1, :]

        v00 = v000 + (v100 - v000) * tx
        v10 = v010 + (v110 - v010) * tx
        v01 = v001 + (v101 - v001) * tx
        v11 = v011 + (v111 - v011) * tx

        v0 = v00 + (v10 - v00) * ty
        v1 = v01 + (v11 - v01) * ty

        out[p, :] = v0 + (v1 - v0) * tz

    if not had_channels:
        return out[:, 0]
    return out


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

    # Construct strictly increasing x per row
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


@pytest.mark.parametrize("uniform_mode", ["auto", "never", "always"])
@pytest.mark.parametrize("outside", ["zero", "clamp"])
def test_interp1d_uniform_modes_on_uniform_grid(uniform_mode: str, outside: str):
    """
    Functional interp1d: uniform='auto'/'always' should work on a uniform x;
    uniform='never' should still match via searchsorted.
    """
    torch.manual_seed(10)
    N, P = 21, 55
    x = torch.linspace(-3.0, 2.0, N, dtype=torch.float32)
    y = torch.randn(N, dtype=torch.float32)

    xq = torch.linspace(
        float(x.min()) - 0.7, float(x.max()) + 0.7, P, dtype=torch.float32
    )

    got = U.interp1d(x, y, xq, outside=outside, uniform=uniform_mode)
    got = _as_1d(got).detach().cpu().numpy()

    want = _numpy_interp_like(x.numpy(), y.numpy(), xq.numpy(), outside=outside)

    assert np.allclose(got, want, atol=2e-5, rtol=1e-4)


def test_interp1d_uniform_always_raises_on_nonuniform_grid():
    x = torch.tensor([0.0, 1.0, 2.2, 3.0], dtype=torch.float32)  # non-uniform spacing
    y = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=torch.float32)
    xq = torch.linspace(-1.0, 4.0, 17, dtype=torch.float32)

    with pytest.raises(ValueError):
        _ = U.interp1d(x, y, xq, outside="clamp", uniform="always")


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

    interp = U.PreparedInterp1d(x, y, outside=outside, sort_xy=True, uniform="auto")

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

    # Strictly increasing x per row via cumulative sum (non-uniform)
    steps = torch.rand(D, N, dtype=torch.float32) + 0.1
    x = torch.cumsum(steps, dim=1)
    y = torch.randn(D, N, dtype=torch.float32)

    # Aligned x_new shape (D,P), includes out-of-bounds
    t = torch.linspace(0.0, 1.0, P, dtype=torch.float32)
    x_min = x[:, :1]
    x_max = x[:, -1:]
    xnew = (x_min - 0.25) + (x_max - x_min + 0.5) * t.unsqueeze(0)

    interp = U.PreparedInterp1d(x, y, outside=outside, sort_xy=True, uniform="auto")
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

    interp = U.PreparedInterp1d(x, y, outside="zero", sort_xy=True, uniform="auto")

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
    interp = U.PreparedInterp1d(x, y, outside=outside, sort_xy=True, uniform="auto")

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


def test_prepared_sort_xy_corrects_unsorted_input_and_uniform_fastpath():
    """
    PreparedInterp1d(sort_xy=True) should work even if x is unsorted by sorting x and permuting y.
    Also validates that uniform='always' can succeed after sorting when the sorted grid is uniform.
    """
    # 1D example: sorted x becomes [0,1,2] (uniform)
    x = torch.tensor([2.0, 0.0, 1.0], dtype=torch.float32)
    y = torch.tensor([20.0, 0.0, 10.0], dtype=torch.float32)
    xq = torch.tensor([-1.0, 0.5, 3.0], dtype=torch.float32)

    interp = U.PreparedInterp1d(x, y, outside="clamp", sort_xy=True, uniform="always")
    assert hasattr(interp, "_use_uniform")
    assert bool(interp._use_uniform) is True

    got = _as_1d(interp(xq)).detach().cpu().numpy()

    xs, perm = torch.sort(x)
    ys = y.index_select(0, perm)
    want = _numpy_interp_like(xs.numpy(), ys.numpy(), xq.numpy(), outside="clamp")

    assert np.allclose(got, want, atol=2e-5, rtol=1e-4)

    # 2D example: each row sorts to a uniform grid
    x2 = torch.tensor([[2.0, 0.0, 1.0], [3.0, 1.0, 2.0]], dtype=torch.float32)
    y2 = torch.tensor([[20.0, 0.0, 10.0], [30.0, 10.0, 20.0]], dtype=torch.float32)
    xq2 = torch.tensor([[0.5, 2.5], [1.5, 3.5]], dtype=torch.float32)

    interp2 = U.PreparedInterp1d(
        x2, y2, outside="clamp", sort_xy=True, uniform="always"
    )
    assert bool(interp2._use_uniform) is True

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

    interp = U.PreparedInterp1d(
        x, y, outside=outside, sort_xy=True, learnable_y=True, uniform="auto"
    )

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


@pytest.mark.parametrize("uniform_mode", ["auto", "never", "always"])
def test_prepared_uniform_modes_on_uniform_grid(uniform_mode: str):
    """
    PreparedInterp1d: verify behavior and internal flag on a uniform grid.
    - auto/always should use uniform path
    - never should force non-uniform path
    """
    torch.manual_seed(6)
    N, P = 33, 77
    x = torch.linspace(-2.0, 5.0, N, dtype=torch.float32)
    y = torch.randn(N, dtype=torch.float32)

    interp = U.PreparedInterp1d(
        x, y, outside="clamp", sort_xy=True, uniform=uniform_mode
    )

    if uniform_mode == "never":
        assert bool(interp._use_uniform) is False
    else:
        assert bool(interp._use_uniform) is True

    xq = torch.linspace(
        float(x.min()) - 0.9, float(x.max()) + 0.9, P, dtype=torch.float32
    )
    got = _as_1d(interp(xq)).detach().cpu().numpy()
    want = _numpy_interp_like(x.numpy(), y.numpy(), xq.numpy(), outside="clamp")

    assert np.allclose(got, want, atol=2e-5, rtol=1e-4)


def test_prepared_uniform_always_raises_on_nonuniform_grid():
    x = torch.tensor([0.0, 1.0, 2.2, 3.0], dtype=torch.float32)
    y = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = U.PreparedInterp1d(x, y, outside="clamp", sort_xy=True, uniform="always")


def test_prepared_fill_value_matches_numpy():
    """
    PreparedInterp1d(outside='fill') should match NumPy's left/right fill behavior.
    """
    torch.manual_seed(7)
    N, P = 11, 31
    fill = 1.2345

    x = torch.linspace(-1.0, 2.0, N, dtype=torch.float32)
    y = torch.randn(N, dtype=torch.float32)

    xq = torch.linspace(
        float(x.min()) - 0.7, float(x.max()) + 0.7, P, dtype=torch.float32
    )

    interp = U.PreparedInterp1d(x, y, outside="fill", fill_value=fill, uniform="auto")
    got = _as_1d(interp(xq)).detach().cpu().numpy()

    want = _numpy_interp_like(
        x.numpy(), y.numpy(), xq.numpy(), outside="fill", fill_value=fill
    )

    assert np.allclose(got, want, atol=2e-5, rtol=1e-4)


# --------------------------------------------------------------------------
# PreparedInterp3dRect (new)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("uniform_mode", ["auto", "never", "always"])
@pytest.mark.parametrize("outside", ["clamp", "zero", "fill"])
def test_prepared3d_unbatched_uniform_matches_numpy(uniform_mode: str, outside: str):
    """
    Unbatched 3D interpolation matches NumPy reference on a uniform grid.
    Also validates uniform fast-path flags for auto/always.
    """
    torch.manual_seed(20)
    Nx, Ny, Nz = 8, 7, 6
    P = 25
    fill = -0.75

    x = torch.linspace(-1.0, 1.0, Nx, dtype=torch.float32)
    y = torch.linspace(0.0, 2.0, Ny, dtype=torch.float32)
    z = torch.linspace(-0.5, 0.5, Nz, dtype=torch.float32)
    values = torch.randn(Nx, Ny, Nz, dtype=torch.float32)

    lo = torch.tensor(
        [float(x.min()) - 0.3, float(y.min()) - 0.3, float(z.min()) - 0.3],
        dtype=torch.float32,
    )
    hi = torch.tensor(
        [float(x.max()) + 0.3, float(y.max()) + 0.3, float(z.max()) + 0.3],
        dtype=torch.float32,
    )
    xyz = lo + (hi - lo) * torch.rand(P, 3, dtype=torch.float32)

    interp = U.PreparedInterp3dRect(
        x,
        y,
        z,
        values,
        outside=outside,
        fill_value=fill,
        sort_xyz=True,
        uniform=uniform_mode,
    )

    # flags should reflect uniform selection
    if uniform_mode == "never":
        assert bool(interp._uniform_x) is False
        assert bool(interp._uniform_y) is False
        assert bool(interp._uniform_z) is False
    else:
        assert bool(interp._uniform_x) is True
        assert bool(interp._uniform_y) is True
        assert bool(interp._uniform_z) is True

    got = interp(xyz).detach().cpu().numpy()

    want = _numpy_trilinear_rect(
        x.numpy(),
        y.numpy(),
        z.numpy(),
        values.numpy(),
        xyz.numpy(),
        outside=outside,
        fill_value=fill,
    )

    assert np.allclose(got, want, atol=2e-4, rtol=1e-4)


def test_prepared3d_unbatched_mixed_uniform_auto_flags_and_correctness():
    """
    Mixed axes: x uniform, y non-uniform, z uniform. uniform='auto' should
    enable fast-path for x/z and disable for y; results must still be correct.
    """
    torch.manual_seed(21)
    Nx, Ny, Nz = 9, 8, 7
    P = 19

    x = torch.linspace(-2.0, 1.0, Nx, dtype=torch.float32)  # uniform
    # non-uniform y via varying steps
    y_steps = torch.linspace(0.05, 0.25, Ny, dtype=torch.float32)
    y = torch.cumsum(y_steps, dim=0)  # strictly increasing, non-uniform
    z = torch.linspace(0.0, 3.0, Nz, dtype=torch.float32)  # uniform

    values = torch.randn(Nx, Ny, Nz, dtype=torch.float32)

    lo = torch.tensor(
        [float(x.min()) - 0.2, float(y.min()) - 0.2, float(z.min()) - 0.2],
        dtype=torch.float32,
    )
    hi = torch.tensor(
        [float(x.max()) + 0.2, float(y.max()) + 0.2, float(z.max()) + 0.2],
        dtype=torch.float32,
    )
    xyz = lo + (hi - lo) * torch.rand(P, 3, dtype=torch.float32)

    interp = U.PreparedInterp3dRect(
        x, y, z, values, outside="clamp", sort_xyz=True, uniform="auto"
    )

    assert bool(interp._uniform_x) is True
    assert bool(interp._uniform_y) is False
    assert bool(interp._uniform_z) is True

    got = interp(xyz).detach().cpu().numpy()
    want = _numpy_trilinear_rect(
        x.numpy(), y.numpy(), z.numpy(), values.numpy(), xyz.numpy(), outside="clamp"
    )

    assert np.allclose(got, want, atol=2e-4, rtol=1e-4)


def test_prepared3d_uniform_always_raises_on_nonuniform_axis():
    Nx, Ny, Nz = 6, 6, 6
    x = torch.tensor([0.0, 0.7, 1.0, 2.1, 2.5, 3.0], dtype=torch.float32)  # non-uniform
    y = torch.linspace(-1.0, 1.0, Ny, dtype=torch.float32)
    z = torch.linspace(0.0, 2.0, Nz, dtype=torch.float32)
    values = torch.randn(Nx, Ny, Nz, dtype=torch.float32)

    with pytest.raises(ValueError):
        _ = U.PreparedInterp3dRect(x, y, z, values, uniform="always")


def test_prepared3d_requires_indices_when_ambiguous():
    torch.manual_seed(22)
    D, Nx, Ny, Nz = 3, 7, 6, 5
    P = 11

    base_x = torch.linspace(-1.0, 1.0, Nx, dtype=torch.float32)
    base_y = torch.linspace(0.0, 2.0, Ny, dtype=torch.float32)
    base_z = torch.linspace(-2.0, -1.0, Nz, dtype=torch.float32)

    x = base_x.unsqueeze(0).expand(D, -1).contiguous()
    y = base_y.unsqueeze(0).expand(D, -1).contiguous()
    z = base_z.unsqueeze(0).expand(D, -1).contiguous()
    values = torch.randn(D, Nx, Ny, Nz, dtype=torch.float32)

    interp = U.PreparedInterp3dRect(x, y, z, values, outside="zero", uniform="auto")

    # xyz_new is (P,3) -> ambiguous in batched mode
    xyz_1 = torch.randn(P, 3, dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = interp(xyz_1)

    # xyz_new is (Q,P,3) with Q != D -> ambiguous without indices
    xyz_2 = torch.randn(D + 1, P, 3, dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = interp(xyz_2)


@pytest.mark.parametrize("outside", ["clamp", "zero", "fill"])
def test_prepared3d_batched_aligned_and_indices_mapping_matches_numpy(outside: str):
    torch.manual_seed(23)
    D, Nx, Ny, Nz = 4, 8, 7, 6
    P = 17
    fill = 2.5

    base_x = torch.linspace(-1.0, 1.0, Nx, dtype=torch.float32)
    base_y = torch.linspace(0.0, 3.0, Ny, dtype=torch.float32)
    base_z = torch.linspace(-2.0, -1.0, Nz, dtype=torch.float32)

    # per-row offsets but still uniform
    x = base_x.unsqueeze(0) + 0.2 * torch.arange(D, dtype=torch.float32).unsqueeze(1)
    y = base_y.unsqueeze(0) + 0.1 * torch.arange(D, dtype=torch.float32).unsqueeze(1)
    z = base_z.unsqueeze(0) + 0.05 * torch.arange(D, dtype=torch.float32).unsqueeze(1)

    values = torch.randn(D, Nx, Ny, Nz, dtype=torch.float32)

    interp = U.PreparedInterp3dRect(
        x,
        y,
        z,
        values,
        outside=outside,
        fill_value=fill,
        sort_xyz=True,
        uniform="auto",
    )

    # (A) aligned query: (D,P,3) with indices=None
    lo = torch.stack([x[:, :1], y[:, :1], z[:, :1]], dim=-1).squeeze(1)  # (D,3)
    hi = torch.stack([x[:, -1:], y[:, -1:], z[:, -1:]], dim=-1).squeeze(1)  # (D,3)
    lo = lo - 0.25
    hi = hi + 0.25

    xyz = lo[:, None, :] + (hi[:, None, :] - lo[:, None, :]) * torch.rand(
        D, P, 3, dtype=torch.float32
    )
    got = interp(xyz)
    assert got.shape == (D, P)

    want = []
    for d in range(D):
        want_d = _numpy_trilinear_rect(
            x[d].numpy(),
            y[d].numpy(),
            z[d].numpy(),
            values[d].numpy(),
            xyz[d].numpy(),
            outside=outside,
            fill_value=fill,
        )
        want.append(want_d)
    want = np.stack(want, axis=0)

    assert np.allclose(got.detach().cpu().numpy(), want, atol=2e-4, rtol=1e-4)

    # (B) indices mapping: (Q,P,3), Q != D
    Q = D + 2
    idx = torch.randint(0, D, (Q,), dtype=torch.long)
    xyzq = torch.randn(Q, P, 3, dtype=torch.float32)

    got2 = interp(xyzq, indices=idx)
    assert got2.shape == (Q, P)

    want2 = []
    for q in range(Q):
        d = int(idx[q])
        want2_q = _numpy_trilinear_rect(
            x[d].numpy(),
            y[d].numpy(),
            z[d].numpy(),
            values[d].numpy(),
            xyzq[q].numpy(),
            outside=outside,
            fill_value=fill,
        )
        want2.append(want2_q)
    want2 = np.stack(want2, axis=0)

    assert np.allclose(got2.detach().cpu().numpy(), want2, atol=2e-4, rtol=1e-4)

    # (C) xyz_new is (P,3) with indices selecting L rows => output (L,P) if L>1
    xyz1 = torch.randn(P, 3, dtype=torch.float32)
    idx3 = torch.tensor([0, 1, 1, 3], dtype=torch.long)
    got3 = interp(xyz1, indices=idx3)
    assert got3.shape == (len(idx3), P)

    want3 = []
    for d in idx3.tolist():
        want3_d = _numpy_trilinear_rect(
            x[d].numpy(),
            y[d].numpy(),
            z[d].numpy(),
            values[d].numpy(),
            xyz1.numpy(),
            outside=outside,
            fill_value=fill,
        )
        want3.append(want3_d)
    want3 = np.stack(want3, axis=0)

    assert np.allclose(got3.detach().cpu().numpy(), want3, atol=2e-4, rtol=1e-4)


def test_prepared3d_channels_output_shape_and_values():
    torch.manual_seed(24)
    Nx, Ny, Nz = 7, 6, 5
    P = 13
    C = 3

    x = torch.linspace(-1.0, 1.0, Nx, dtype=torch.float32)
    y = torch.linspace(0.0, 2.0, Ny, dtype=torch.float32)
    z = torch.linspace(-0.5, 0.5, Nz, dtype=torch.float32)
    values = torch.randn(Nx, Ny, Nz, C, dtype=torch.float32)

    xyz = torch.randn(P, 3, dtype=torch.float32)  # includes OOB
    interp = U.PreparedInterp3dRect(x, y, z, values, outside="clamp", uniform="auto")

    got = interp(xyz)
    assert got.shape == (P, C)

    want = _numpy_trilinear_rect(
        x.numpy(), y.numpy(), z.numpy(), values.numpy(), xyz.numpy(), outside="clamp"
    )

    assert np.allclose(got.detach().cpu().numpy(), want, atol=2e-4, rtol=1e-4)


def test_prepared3d_learnable_values_grad():
    torch.manual_seed(25)
    Nx, Ny, Nz = 6, 5, 4
    P = 17

    x = torch.linspace(-1.0, 1.0, Nx, dtype=torch.float32)
    y = torch.linspace(0.0, 2.0, Ny, dtype=torch.float32)
    z = torch.linspace(-0.5, 0.5, Nz, dtype=torch.float32)
    values = torch.randn(Nx, Ny, Nz, dtype=torch.float32)

    interp = U.PreparedInterp3dRect(
        x,
        y,
        z,
        values,
        outside="clamp",
        learnable_values=True,
        uniform="auto",
    )

    xyz = torch.randn(P, 3, dtype=torch.float32)
    out = interp(xyz)
    loss = out.pow(2).mean()
    loss.backward()

    assert hasattr(interp, "values")
    assert interp.values.grad is not None
    assert float(interp.values.grad.abs().sum()) > 0.0
