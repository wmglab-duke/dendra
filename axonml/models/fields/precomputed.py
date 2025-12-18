"""Precomputed fields."""

import glob

import numpy as np
import torch
from natsort import natsorted

from axonml.utils import PreparedInterp1d

from .quasipotentials import calculate_quasipotentials_batched_coords


class PreComputedInterpolate1D(torch.nn.Module):
    """Use PreComputed data sampled along x at different (y, z) locations."""

    def __init__(
        self,
        data=None,
        x=None,
        outside="zero",
        *,
        truncate: float | None = None,
        truncate_mode: str = "best_fit",  # "best_fit" or "safe"
        **kwargs,
    ):
        super().__init__()
        self.outside = outside

        # Convert inputs to tensors (keep on CPU initially; moved in forward via self.to)
        if data is not None and not torch.is_tensor(data):
            data = torch.as_tensor(data)
        if x is not None and not torch.is_tensor(x):
            x = torch.as_tensor(x)

        # Force 2D (this class conceptually represents a batch of LUT rows)
        if data is None or x is None:
            raise ValueError("data and x must be provided.")
        if data.ndim == 1:
            data = data.unsqueeze(0)  # (1,N)
        if x.ndim == 1:
            # Expand shared x to match data rows (no memory copy)
            x = x.unsqueeze(0).expand(data.shape[0], -1)  # (D,N)
        if data.ndim != 2 or x.ndim != 2:
            raise ValueError("data and x must be 2D tensors after normalization.")
        if data.shape != x.shape:
            raise ValueError(
                f"data and x must have identical shape; got data {tuple(data.shape)} vs x {tuple(x.shape)}"
            )

        # We'll manage sorting ourselves so truncation is well-defined
        sort_xy = bool(kwargs.pop("sort_xy", True))
        if sort_xy:
            x, data = self._sort_xy_2d(x, data)
        else:
            # sanity: monotonic increasing is required
            if not torch.all(x[:, 1:] >= x[:, :-1]):
                raise ValueError(
                    "sort_xy=False but x is not sorted ascending along the last dim."
                )

        # Optional symmetric truncation
        if truncate is not None and float(truncate) != 0.0:
            x, data, (k_left, k_right) = self._truncate_xy_2d(
                x, data, truncate=float(truncate), mode=truncate_mode
            )
            self.truncate = float(truncate)
            self.truncate_mode = str(truncate_mode)
            self._truncate_left = int(k_left)
            self._truncate_right = int(k_right)
        else:
            self.truncate = 0.0
            self.truncate_mode = str(truncate_mode)
            self._truncate_left = 0
            self._truncate_right = 0

        # Outside mapping: point_source uses base interpolation with zero outside,
        # then we fill outside with point-source model.
        if outside == "point_source":
            outside_for_interpolation = "zero"
        else:
            outside_for_interpolation = outside

        # Build PreparedInterp1d with already-sorted + truncated data
        self.interp = PreparedInterp1d(
            x=x,
            y=data,
            outside=outside_for_interpolation,
            sort_xy=False,  # already sorted
            **kwargs,
        )

    @staticmethod
    def _sort_xy_2d(
        x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sort x row-wise and permute y identically."""
        xs, perm = torch.sort(x, dim=1)
        ys = torch.gather(y, dim=1, index=perm)
        return xs, ys

    @staticmethod
    def _normalize_truncate(truncate: float) -> float:
        """
        Accept either:
          - fraction in (0,1): e.g. 0.1 => 10%
          - percent in [1,100): e.g. 10 => 10%
        """
        t = float(truncate)
        if t <= 0:
            return 0.0
        if 0 < t < 1:
            return t
        if 1 <= t < 100:
            return t / 100.0
        raise ValueError(
            "truncate must be in (0,1) as a fraction or in [1,100) as a percent."
        )

    @classmethod
    def _truncate_xy_2d(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        truncate: float,
        mode: str = "best_fit",
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """
        Symmetrically truncate total fraction `truncate` from the x-domain:
          - remove truncate/2 on the left and truncate/2 on the right (in x-range units)

        Works for x,y shaped (D,N). Returns (x_trunc, y_trunc, (k_left, k_right)).

        mode:
          - "best_fit": choose uniform k_left / k_right to minimize mean abs error
                        in realized truncation fraction across rows.
          - "safe": choose uniform k_left / k_right so that every row truncates at
                    least the requested fraction (may over-truncate).
        """
        if mode not in {"best_fit", "safe"}:
            raise ValueError("truncate_mode must be 'best_fit' or 'safe'")

        t = cls._normalize_truncate(truncate)
        if t <= 0.0:
            return x, y, (0, 0)
        if t >= 1.0:
            raise ValueError("truncate must be < 1.0 (i.e., < 100%).")

        D, N = x.shape
        if N < 3:
            raise ValueError("Need N>=3 to truncate while leaving at least 2 points.")

        frac_each = 0.5 * t  # e.g. truncate=0.1 => 0.05 each side
        device = x.device
        dtype = x.dtype
        eps = torch.finfo(dtype).eps * 10.0

        x_min = x[:, :1]  # (D,1)
        x_max = x[:, -1:]  # (D,1)
        rng = x_max - x_min  # (D,1)
        if not torch.all(rng > 0):
            raise ValueError("x must be strictly increasing per row (x_max > x_min).")

        target = torch.tensor(frac_each, device=device, dtype=dtype)

        if mode == "safe":
            # Thresholds in x-units
            left_thr = x_min + rng * target
            right_thr = x_max - rng * target

            # searchsorted row-wise
            # left: first index with x >= left_thr
            k_left_per = torch.searchsorted(x, left_thr, right=False).squeeze(1)  # (D,)
            # right: exclusive index after last x <= right_thr
            right_idx_per = torch.searchsorted(x, right_thr, right=True).squeeze(
                1
            )  # (D,)
            k_right_per = N - right_idx_per  # (D,)

            # Choose uniform truncation that satisfies all rows (at least requested)
            k_left = int(k_left_per.max().item())
            k_right = int(k_right_per.max().item())

        else:
            # "best_fit": choose k_left and k_right that minimize mean |realized - target|
            # Candidate k in [0, N-2] (must leave at least 2 points overall after both cuts;
            # we enforce that after choosing).
            cand = torch.arange(0, N - 1, device=device, dtype=torch.long)  # length N-1

            # Left realized fraction if we start at index k: (x[:,k] - x_min)/rng
            u_left = (x - x_min) / rng  # (D,N)
            f_left_cand = u_left[:, cand]  # (D,N-1)
            err_left = (f_left_cand - target).abs().mean(dim=0)  # (N-1,)
            k_left = int(err_left.argmin().item())

            # Right realized fraction if we drop k points: (x_max - x[:,N-1-k])/rng
            idx_last = N - 1 - cand  # (N-1,)
            x_last = x.index_select(1, idx_last)  # (D,N-1)
            f_right_cand = (x_max - x_last) / rng  # (D,N-1)
            err_right = (f_right_cand - target).abs().mean(dim=0)
            k_right = int(err_right.argmin().item())

        # Enforce at least 2 points kept
        if (N - k_left - k_right) < 2:
            # Reduce truncation conservatively
            # Keep exactly 2 points: set k_right to max allowed given k_left
            k_right = max(0, N - k_left - 2)

        # Slice uniformly
        x_t = x[:, k_left : N - k_right]
        y_t = y[:, k_left : N - k_right]

        if x_t.shape[1] < 2:
            raise ValueError(
                "Truncation removed too many points; need at least 2 points remaining."
            )

        # Extra sanity: preserve monotonicity
        if not torch.all(x_t[:, 1:] >= x_t[:, :-1] - eps):
            raise ValueError(
                "Truncation produced a non-monotone x; check x sorting and truncate parameters."
            )

        return x_t, y_t, (k_left, k_right)

    @classmethod
    def from_ascent(cls, ascent_dir, sample, model, sim, contact, **kwargs):
        data = glob.glob(
            f"{ascent_dir}/samples/{sample}/models/{model}/sims/{sim}/fibersets_bases/0/{contact}/*.dat"
        )
        data = natsorted(data)
        data = np.vstack([np.loadtxt(f, skiprows=1) for f in data])

        x = np.loadtxt(
            f"{ascent_dir}/samples/{sample}/models/{model}/sims/{sim}/fibersets/0/0.dat",
            skiprows=1,
        )[:, -1]

        return cls(data=data, x=x, **kwargs)

    def _normalize_indices(
        self, indices, Q: int, device: torch.device
    ) -> torch.Tensor | None:
        """
        Normalize indices to a length-Q LongTensor, or None.

        If indices has length L and Q is a multiple of L, we tile indices to length Q.
        This matches the original intention of repeating indices, but fixes the factor.
        """
        if indices is None:
            return None

        idx = torch.as_tensor(indices, dtype=torch.long, device=device).reshape(-1)

        if idx.numel() == 1:
            return idx.expand(Q)

        if idx.numel() == Q:
            return idx

        if Q % idx.numel() == 0:
            reps = Q // idx.numel()
            return idx.repeat(reps)

        raise ValueError(
            f"indices length {idx.numel()} is not 1 and does not match Q={Q} (and does not tile cleanly)."
        )

    def _interp_tables_for_rows(
        self, Q: int, idx: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (x_tab, y_tab) with shape (Q,N) each, selecting LUT rows if needed.
        """
        # x table
        if hasattr(self.interp, "_x_search"):
            x_tab = self.interp._x_search  # (N,) or (D,N)
        elif hasattr(self.interp, "x"):
            x_tab = self.interp.x
        else:
            raise AttributeError(
                "PreparedInterp1d does not expose x table (_x_search or x)."
            )

        # y table
        if hasattr(self.interp, "y"):  # learnable_y=True
            y_tab = self.interp.y
        elif hasattr(self.interp, "_y"):
            y_tab = self.interp._y
        else:
            raise AttributeError("PreparedInterp1d does not expose y table (y or _y).")

        # Select rows if tables are batched and indices provided
        if x_tab.ndim == 2 and idx is not None:
            x_tab = x_tab.index_select(0, idx)
        if y_tab.ndim == 2 and idx is not None:
            y_tab = y_tab.index_select(0, idx)

        # Broadcast/validate to (Q,N)
        if x_tab.ndim == 1:
            x_tab = x_tab.unsqueeze(0).expand(Q, -1)
        else:
            if x_tab.shape[0] != Q:
                raise ValueError(
                    f"x table has {x_tab.shape[0]} rows but expected Q={Q}. Provide correct indices."
                )

        if y_tab.ndim == 1:
            y_tab = y_tab.unsqueeze(0).expand(Q, -1)
        else:
            if y_tab.shape[0] != Q:
                raise ValueError(
                    f"y table has {y_tab.shape[0]} rows but expected Q={Q}. Provide correct indices."
                )

        return x_tab, y_tab

    @staticmethod
    def _solve_point_source_d(
        peak_x: torch.Tensor,
        peak_v: torch.Tensor,
        end_x: torch.Tensor,
        end_v: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve for d from:
          V_end = V_peak * d / sqrt(d^2 + (end_x - peak_x)^2)

        Returns:
          d     : (Q,)
          valid : (Q,) boolean
        """
        peak_mag = peak_v.abs()
        end_mag = end_v.abs()

        xdiff = end_x - peak_x  # (Q,)
        ratio2 = (peak_mag / (end_mag + eps)).pow(2)  # (Q,)
        den = ratio2 - 1.0

        # Valid when:
        # - endpoint nonzero
        # - peak larger than endpoint
        # - den > 0
        # - xdiff nonzero (peak not exactly at boundary)
        valid = (
            (end_mag > eps)
            & (peak_mag > (end_mag + eps))
            & (den > eps)
            & (xdiff.abs() > eps)
        )

        d2 = xdiff.pow(2) / torch.clamp(den, min=eps)  # safe
        d = torch.sqrt(torch.clamp(d2, min=0.0))

        return d, valid

    def point_source_fill(
        self,
        x_vec: torch.Tensor,  # (Q,P)
        interp: torch.Tensor,  # (Q,P) from PreparedInterp1d with outside="zero"
        *,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fill out-of-bounds values (x < x_min or x > x_max) using a point-source extrapolation.

        This is done row-wise, using the LUT row to determine:
        - peak position/value
        - boundary values at x_min and x_max

        Returns a new tensor (no in-place writes) to remain autograd-friendly.
        """
        if x_vec.ndim != 2 or interp.ndim != 2:
            raise ValueError(
                "point_source_fill expects x_vec and interp as 2D tensors (Q,P)."
            )
        if x_vec.shape != interp.shape:
            raise ValueError(
                f"x_vec shape {tuple(x_vec.shape)} must match interp shape {tuple(interp.shape)}."
            )

        Q, P = x_vec.shape
        dtype = interp.dtype
        finfo = torch.finfo(dtype)
        eps = float(finfo.eps) * 10.0  # modestly larger than eps for stability

        # LUT tables per row
        x_tab, y_tab = self._interp_tables_for_rows(Q=Q, idx=indices)  # (Q,N), (Q,N)

        x_min = x_tab[:, 0]  # (Q,)
        x_max = x_tab[:, -1]  # (Q,)

        # Determine peak (by magnitude) from the LUT, not from query samples
        peak_idx = y_tab.abs().argmax(dim=1)  # (Q,)
        peak_x = x_tab.gather(1, peak_idx[:, None]).squeeze(1)  # (Q,)
        peak_v = y_tab.gather(1, peak_idx[:, None]).squeeze(1)  # (Q,)

        # Boundary values from LUT endpoints
        end_x_left = x_min
        end_v_left = y_tab[:, 0]
        end_x_right = x_max
        end_v_right = y_tab[:, -1]

        # Solve d separately for each side
        d_left, valid_left = self._solve_point_source_d(
            peak_x, peak_v, end_x_left, end_v_left, eps
        )
        d_right, valid_right = self._solve_point_source_d(
            peak_x, peak_v, end_x_right, end_v_right, eps
        )

        # If only one side is solvable, use it for both sides (symmetric point-source assumption)
        valid_any = valid_left | valid_right  # (Q,)
        d_left_eff = torch.where(valid_left, d_left, d_right)
        d_right_eff = torch.where(valid_right, d_right, d_left)

        # Masks of query points outside the interpolator bounds
        left_mask = (x_vec < x_min[:, None]) & valid_any[:, None]
        right_mask = (x_vec > x_max[:, None]) & valid_any[:, None]

        # If nothing to fill, return original
        if not (left_mask.any() or right_mask.any()):
            return interp

        # Point-source potential using peak_v and d:
        # V(x) = V_peak * d / sqrt(d^2 + (x - peak_x)^2)
        dist = x_vec - peak_x[:, None]  # (Q,P)

        # Left fill
        r_left = torch.sqrt(d_left_eff[:, None].pow(2) + dist.pow(2))
        fill_left = peak_v[:, None] * d_left_eff[:, None] / torch.clamp(r_left, min=eps)

        # Right fill
        r_right = torch.sqrt(d_right_eff[:, None].pow(2) + dist.pow(2))
        fill_right = (
            peak_v[:, None] * d_right_eff[:, None] / torch.clamp(r_right, min=eps)
        )

        # Apply fills outside domain only
        out = interp
        if left_mask.any():
            out = torch.where(left_mask, fill_left, out)
        if right_mask.any():
            out = torch.where(right_mask, fill_right, out)

        return out

    def forward(self, model, indices=None):
        x = model.x
        self.to(device=x.device, dtype=x.dtype)

        shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])  # (Q,P)
        Q = x_2d.shape[0]

        idx = self._normalize_indices(indices, Q=Q, device=x.device)

        # Call PreparedInterp1d correctly (indices is keyword-only in the newer implementation)
        if idx is None:
            interpolated = self.interp(x_2d)
        else:
            interpolated = self.interp(x_2d, indices=idx)

        # Point-source extrapolation is applied only outside the bounds.
        if self.outside == "point_source":
            interpolated = self.point_source_fill(x_2d, interpolated, indices=idx)

        return interpolated.reshape(shape)


# -- aliases --
FEMInterpolate1D = PreComputedInterpolate1D


class EfieldInterpolate3D(torch.nn.Module):
    def __init__(
        self,
        xyz,
        efield,
        *,
        k: int | None = 8,
        eps: float = 1e-9,
        chunksize: int = None,
    ):
        """
        Initialize the EfieldInterpolate3D module.

        Parameters
        ----------
        xyz : torch.Tensor
            A tensor of shape (N, 3) containing the coordinates (x, y, z) in μm.
        efield : torch.Tensor
            A tensor of shape (N, 3) containing the electric field vectors at the coordinates.
        k : int, optional
            The number of nearest neighbors to consider for interpolation. Default is 8.
        eps : float, optional
            A small value to avoid division by zero in interpolation. Default is 1e-9.

        Forward
        -------
        forward(x, y, z) → (B, K, 3) tensor
            `x`, `y`, `z` are each (B, K) tensors of coordinates.  The output
            is the interpolated E-field at every query point using
            inverse-distance weighting.

        Notes
        -----
        *  All operations remain on the same device/dtype as the inputs.
        *  The module is differentiable w.r.t. *query* coordinates; the
           sample points/values are treated as constants (buffers).
        """
        super().__init__()
        assert xyz.shape == efield.shape and xyz.shape[1] == 3
        N = xyz.shape[0]
        if k is not None and (k < 1 or k > N):
            raise ValueError(f"k must be in [1, N={N}] or None.")
        if chunksize is not None and (chunksize < 1 or chunksize > N):
            raise ValueError(f"chunksize must be in [1, N={N}] or None.")

        xyz = torch.as_tensor(xyz)
        efield = torch.as_tensor(efield)

        # store as buffers so they move with .to(device) / .half() calls
        self.register_buffer("xyz", xyz.clone())
        self.register_buffer("efield", efield.clone())
        self.k = k
        self.eps = eps
        self.chunksize = chunksize or N

    # ------------------------------------------------------------------
    # core helper: inverse‑distance weighting on last dim
    # ------------------------------------------------------------------
    def _idw(self, dist2: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
        """
        dist2 : (..., M) squared distances
        vecs  : (..., M, 3) corresponding vectors
        Returns
        -------
        (..., 3) weighted average
        """
        w = 1.0 / (dist2 + self.eps)
        w = w / w.sum(dim=-1, keepdim=True)
        return (w.unsqueeze(-1) * vecs).sum(dim=-2)

    # ------------------------------------------------------------------
    # user‑facing API
    # ------------------------------------------------------------------
    def __interp(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        x, y, z : (B, K) query coordinates
        Returns  (B, K, 3) interpolated E-field
        """
        if x.shape != y.shape or x.shape != z.shape:
            raise ValueError("x, y, z must have identical shapes (B, K)")

        B, K = x.shape
        xyz_q = torch.stack((x, y, z), dim=-1)  # (B, K, 3)

        # ---------- pair‑wise squared distances ----------
        #   diff → (B, K, N, 3)
        diff = xyz_q[..., None, :] - self.xyz  # broadcast N
        dist2 = (diff**2).sum(dim=-1)  # (B, K, N)

        # ---------- pick k nearest neighbours if requested ----------
        if self.k is not None and self.k < self.xyz.shape[0]:
            dist2, idx = torch.topk(dist2, self.k, dim=-1, largest=False)
            vecs = self.efield[idx]  # (B, K, k, 3)
        else:  # use all N
            vecs = self.efield.expand(B, K, -1, -1)  # broadcast to (B, K, N, 3)

        # ---------- inverse‑distance weighted average ----------
        return self._idw(dist2, vecs)

    def _knn_chunked(self, xyz_q, k, chunk=32768):
        # xyz_q : (Q, 3) where Q = B*K
        Q = xyz_q.size(0)
        best_dist2 = torch.full((Q, k), float("inf"), device=xyz_q.device)
        best_idx = torch.full((Q, k), -1, dtype=torch.long, device=xyz_q.device)

        for start in range(0, self.xyz.size(0), chunk):
            end = min(start + chunk, self.xyz.size(0))
            src = self.xyz[start:end]  # (chunk, 3)
            dist2 = ((xyz_q[:, None, :] - src[None]) ** 2).sum(-1)  # (Q, chunk)

            # concatenate current best with this block, then keep k smallest
            dist2_cat = torch.cat((best_dist2, dist2), dim=1)  # (Q, k+chunk)
            idx_cat = torch.cat(
                (best_idx, torch.arange(start, end, device=xyz_q.device).expand(Q, -1)),
                dim=1,
            )  # (Q, k+chunk)

            best_dist2, sel = torch.topk(dist2_cat, k, dim=1, largest=False)
            best_idx = idx_cat.gather(1, sel)

        return best_dist2, best_idx  # (Q, k), (Q, k)

    def _interp(self, x, y, z):
        B, K = x.shape
        xyz_q = torch.stack((x, y, z), dim=-1).reshape(-1, 3)  # (Q, 3)

        dist2, idx = self._knn_chunked(xyz_q, self.k, self.chunksize)
        vecs = self.efield[idx]  # (Q, k, 3)
        efield = self._idw(dist2, vecs).view(B, K, 3)
        return efield

    def forward(self, model):
        self.to(device=model.device(), dtype=model.dtype())
        x, y, z = model.x, model.y, model.z
        G = model.graph
        efield = self._interp(x, y, z)
        return calculate_quasipotentials_batched_coords(G, x, y, z, efield)
