"""Precomputed fields."""

import glob

import numpy as np
import torch
from natsort import natsorted

from dendra.helpers import logger
from dendra.utils import (
    PreparedInterp1d,
    PreparedInterp3dFEM,
    PreparedInterp3dRect,
    PreparedInterp3dScattered,
)

from .quasipotentials import calculate_quasipotentials_batched_coords


class PreComputedInterpolate1D(torch.nn.Module):
    """
    Use PreComputed data sampled along x at different (y, z) locations.

    Parameters
    ----------
    data : torch.Tensor
        2D tensor of shape (D, N) containing the precomputed field values.  The
        values retain the caller's normalization; see Notes.
    x : torch.Tensor
        1D or 2D tensor of shape (N,) or (D, N) containing the x-coordinates
        in μm corresponding to the data values.  If 1D, the same x-coordinates
        are used for all D rows. If 2D, each row can have different
        x-coordinates (i.e., can be sampled at different locations along x).
    outside : str, optional
        Behavior for points outside the interpolation range. Options are "zero" or "point_source".
        If "point_source", the interpolation uses a point-source model outside the (optionally
        truncated) support. Default is "zero".
    truncate : float or None, optional
        Amount to truncate the data symmetrically. Default is None (no truncation).
    truncate_mode : str, optional
        Mode for truncation, either "best_fit" or "safe". Default is "best_fit".
    **kwargs
        Additional keyword arguments passed to :class:`PreparedInterp1d`.

    Notes
    -----
    - The input `data` and `x` tensors must have the same shape (D, N), where D is the
      number of data rows and N is the number of samples per row.
    - The `x` values for each row must be sorted in ascending order, either by setting
      `sort_xy=True` (default) or ensuring they are pre-sorted when `sort_xy=False`.
    - Truncation removes a fraction of the data symmetrically from both ends of the x-domain.
    - The `outside` parameter determines how values outside the interpolation range are handled.
      If set to "point_source", a point-source extrapolation is applied based on the peak
      and boundary values of each row.
    - Interpolation preserves the units and reference-amplitude normalization of
      ``data``.  For extracellular stimulation, ``data`` may be an absolute
      potential in mV paired with a dimensionless waveform, or a lead field in
      mV per input unit paired with a waveform in that unit.  The final product
      supplied through ``extra`` must be in mV.
    - Model coordinates follow the field contract ``(*batch, N, C)``. A single
      LUT row broadcasts everywhere. A bank of ``N`` rows aligns with the model
      population axis and repeats over every outer batch axis. A bank with one
      row per flattened logical lane also aligns row-wise. Other mappings are
      ambiguous and require explicit ``indices``.
    """

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
        """
        Create a PreComputedInterpolate1D instance from precomputed data generated by the ASCENT pipeline.

        Parameters
        ----------
        ascent_dir : str
            "ASCENT_PROJECT_PATH" as used in your ASCENT env.json, e.g. "/path/to/ascent_project".
        sample : int
            Sample number, e.g. 0.
        model : int
            Model number, e.g. 0.
        sim : int
            Simulation number, e.g. 0.
        contact : int
            Contact number, e.g. 0.

        Returns
        -------
        PreComputedInterpolate1D
            An instance initialized with the precomputed data converted from V
            to mV.  This conversion preserves the ASCENT dataset's original
            reference-amplitude normalization.
        """
        data = glob.glob(
            f"{ascent_dir}/samples/{sample}/models/{model}/sims/{sim}/fibersets_bases/0/{contact}/*.dat"
        )
        data = natsorted(data)
        data = [np.loadtxt(f, skiprows=1).flatten() for f in data]

        x_data = glob.glob(
            f"{ascent_dir}/samples/{sample}/models/{model}/sims/{sim}/fibersets/0/*.dat"
        )
        x_data = natsorted(x_data)
        x_data = [np.loadtxt(f, skiprows=1)[:, -1].flatten() for f in x_data]

        if not data or not x_data:
            raise ValueError("ASCENT field and coordinate files must both be present.")
        if len(data) != len(x_data):
            raise ValueError("Number of data files and x files must match.")
        if not all(d.shape[0] == x.shape[0] for d, x in zip(data, x_data)):
            raise ValueError(
                "Each data file must have the same number of rows as its "
                "corresponding x file."
            )

        if not all(d.shape[0] == data[0].shape[0] for d in data):
            # we need to resample to a common number of points (e.g. max) for the batch interpolation; we'll let the class handle that with interpolation
            logger.info(
                "Data files have varying number of points; relying on interpolation to handle this."
            )
            n_resampled = int(np.median([d.shape[0] for d in data]))
            data_resampled = []
            x_data_resampled = []
            for d, x in zip(data, x_data):
                x_tensor = torch.as_tensor(x, dtype=torch.float64)
                data_tensor = torch.as_tensor(d, dtype=x_tensor.dtype)
                interp = PreparedInterp1d(x=x_tensor, y=data_tensor)
                x_resampled = torch.linspace(
                    x_tensor.min(),
                    x_tensor.max(),
                    n_resampled,
                    dtype=x_tensor.dtype,
                )
                d_resampled = interp(x_resampled)
                data_resampled.append(d_resampled.flatten().cpu().numpy())
                x_data_resampled.append(x_resampled.flatten().cpu().numpy())
            data = data_resampled
            x_data = x_data_resampled

        data = np.vstack(data) * 1000  # (D,N), ASCENT stores in V so convert to mV
        x_data = np.vstack(x_data)  # (D,N)
        return cls(data=data, x=x_data, **kwargs)

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

        if idx.numel() == 0:
            raise ValueError("indices must contain at least one LUT-row index.")

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

    def _point_source_fill(
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
        """
        Interpolate field at model.x using precomputed 1D LUT.

        Parameters
        ----------
        model : dendra.models.Population
            Model with .x coordinates where field is evaluated.
        indices : array-like of int, optional
            Indices selecting the LUT row for each flattened logical model lane.
            A scalar broadcasts, a length-``Q`` vector maps every lane, and a
            shorter vector that tiles evenly is repeated in flattened row-major
            lane order. If omitted, the automatic mappings described in Notes
            apply.
        """
        x = model.x
        self.to(device=x.device, dtype=x.dtype)

        shape = x.shape
        if x.ndim < 2:
            raise ValueError(
                "model.x must have shape (*batch, N, C), including population "
                "and compartment axes."
            )
        x_2d = x.reshape(-1, x.shape[-1])  # (Q,P)
        Q = x_2d.shape[0]

        idx = self._normalize_indices(indices, Q=Q, device=x.device)
        if idx is None and self.interp.batched:
            n_tables = self.interp.D
            n_population = x.shape[-2]
            if n_tables == 1:
                # PreComputedInterpolate1D stores even a shared table as (1,S),
                # so explicitly select row zero for every flattened lane.
                idx = torch.zeros(Q, dtype=torch.long, device=x.device)
            elif n_tables == Q:
                # PreparedInterp1d's implicit row-wise mapping is exact.
                idx = None
            elif n_tables == n_population:
                # Flattening (*batch,N,C) makes N the fastest-changing logical
                # axis, so repeat the population LUT bank for each outer batch.
                idx = torch.arange(n_tables, dtype=torch.long, device=x.device)
                idx = idx.repeat(Q // n_population)
            else:
                raise ValueError(
                    "Cannot infer a PreComputedInterpolate1D LUT-row mapping: "
                    f"the table bank has D={n_tables} rows, while model.x has "
                    f"shape {tuple(shape)} (Q={Q} flattened logical lanes and "
                    f"N={n_population} population rows). Automatic mapping "
                    "requires D == 1, D == Q, or D == N; provide explicit "
                    "indices otherwise."
                )

        # Call PreparedInterp1d correctly (indices is keyword-only in the newer implementation)
        if idx is None:
            interpolated = self.interp(x_2d)
        else:
            interpolated = self.interp(x_2d, indices=idx)

        # Point-source extrapolation is applied only outside the bounds.
        if self.outside == "point_source":
            interpolated = self._point_source_fill(x_2d, interpolated, indices=idx)

        return interpolated.reshape(shape)


class PreComputedInterpolate3DRect(torch.nn.Module):
    """Interpolate 3D field from rectilinear grid.

    Parameters
    ----------
    x : torch.Tensor
        1D tensor of x-coordinates of the grid points (in μm).
    y : torch.Tensor
        1D tensor of y-coordinates of the grid points (in μm).
    z : torch.Tensor
        1D tensor of z-coordinates of the grid points (in μm).
    field : torch.Tensor
        Tensor of shape (Nx, Ny, Nz, ...) containing the field values at the
        grid points.  Interpolation preserves the caller's units and
        reference-amplitude normalization.
    **kwargs : additional keyword arguments for PreparedInterp3dRect.

    Notes
    -----
    *  Model coordinates and the scalar result have shape ``(*batch, N, C)``.
    *  All operations remain on the same device/dtype as the model.
    *  The module is differentiable w.r.t. the model's coordinates; the
       field grid points/values are treated as constants (buffers).
    *  The module is differentiable w.r.t. the field values, but must be
       specified explicitly with learnable=True as a kwarg.
    *  For extracellular stimulation, the interpolated field multiplied by its
       temporal waveform must be in mV.
    """

    def __init__(self, x, y, z, field, **kwargs):
        super().__init__()
        self.interpolator = PreparedInterp3dRect(x=x, y=y, z=z, values=field, **kwargs)

    def _interp(self, x, y, z):
        shape = x.shape
        xyz_q = torch.stack((x, y, z), dim=-1).reshape(-1, 3)  # (Q, 3)
        field = self.interpolator(xyz_q).view(*shape)
        return field

    def forward(self, model):
        """
        Interpolate field at model.x, model.y, model.z using precomputed 3D rectilinear grid.

        Parameters
        ----------
        model : dendra.models.Population
            Model with .x, .y, .z coordinates where field is evaluated.

        Returns
        -------
        field : torch.Tensor
            Tensor with the same ``(*batch, N, C)`` shape as model coordinates.
        """
        self.to(device=model.device(), dtype=model.dtype())
        x, y, z = model.x, model.y, model.z
        field = self._interp(x, y, z)
        return field


class PreComputedInterpolate3DScattered(torch.nn.Module):
    """Interpolate 3D field from scattered points.

    Parameters
    ----------
    xyz : torch.Tensor
        A tensor of shape (N, 3) containing the coordinates (x, y, z) in μm.
    field : torch.Tensor
        A tensor of shape (N, 1) containing the field values at the coordinates.
        Interpolation preserves the caller's units and reference-amplitude
        normalization.
    **kwargs : additional keyword arguments for PreparedInterp3dScattered.

    Notes
    -----
    *  Model coordinates and the scalar result have shape ``(*batch, N, C)``.
    *  All operations remain on the same device/dtype as the model.
    *  The module is differentiable w.r.t. the model's coordinates; the
       sample points/values are treated as constants (buffers).
    *  The module is differentiable w.r.t. the field values, but must be
       specified explicitly with learnable=True as a kwarg.
    *  For extracellular stimulation, the interpolated field multiplied by its
       temporal waveform must be in mV.
    """

    def __init__(
        self,
        xyz,
        field,
        **kwargs,
    ):
        super().__init__()
        xyz = torch.as_tensor(xyz)
        field = torch.as_tensor(field)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3).")
        if field.ndim != 2 or field.shape[1] != 1:
            raise ValueError("field must have shape (N, 1).")
        if xyz.shape[0] != field.shape[0]:
            raise ValueError("xyz and field must contain the same number of samples.")

        self.interpolator = PreparedInterp3dScattered(
            points=xyz,
            values=field,
            **kwargs,
        )

    def _interp(self, x, y, z):
        shape = x.shape
        xyz_q = torch.stack((x, y, z), dim=-1).reshape(-1, 3)  # (Q, 3)
        field = self.interpolator(xyz_q).view(*shape)
        return field

    def forward(self, model):
        """
        Interpolate field at model.x, model.y, model.z using precomputed 3D scattered points.

        Parameters
        ----------
        model : dendra.models.Population
            Model with .x, .y, .z coordinates where field is evaluated.

        Returns
        -------
        field : torch.Tensor
            Tensor with the same ``(*batch, N, C)`` shape as model coordinates.
        """
        self.to(device=model.device(), dtype=model.dtype())
        x, y, z = model.x, model.y, model.z
        field = self._interp(x, y, z)
        return field


class _MeshCoordinateTransformMixin:
    """Shared coordinate handling for mesh-backed interpolators."""

    def _init_mesh_coordinate_transform(
        self,
        *,
        coordinate_scale=1.0,
        coordinate_offset=None,
    ):
        self.register_buffer(
            "_coordinate_scale",
            torch.as_tensor(float(coordinate_scale), dtype=torch.float32),
        )
        if coordinate_offset is None:
            self.register_buffer("_coordinate_offset", None)
        else:
            offset = torch.as_tensor(coordinate_offset, dtype=torch.float32)
            if offset.shape != (3,):
                raise ValueError("coordinate_offset must be None or a length-3 vector.")
            self.register_buffer("_coordinate_offset", offset)

    def _query_xyz_from_model_coords(self, x, y, z):
        xyz_q = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        scale = self._coordinate_scale.to(device=xyz_q.device, dtype=xyz_q.dtype)
        xyz_q = xyz_q * scale
        if self._coordinate_offset is not None:
            offset = self._coordinate_offset.to(device=xyz_q.device, dtype=xyz_q.dtype)
            xyz_q = xyz_q + offset
        return xyz_q


class PreComputedInterpolate3DMesh(_MeshCoordinateTransformMixin, torch.nn.Module):
    """Interpolate a scalar mesh-defined field at model coordinates.

    This wrapper is the tetrahedral-mesh analogue of
    :class:`PreComputedInterpolate3DRect` and
    :class:`PreComputedInterpolate3DScattered`.  It delegates the actual
    mesh-guided FEM interpolation to :class:`PreparedInterpolate3dFEM`, which
    reproduces SimNIBS ``NodeData.interpolate_scattered`` behavior.

    Parameters
    ----------
    interpolator : PreparedInterpolate3dFEM
        Prepared FEM interpolator, normally created with :meth:`from_NodeData`.
    coordinate_scale : float, optional
        Multiplicative conversion from model coordinates to mesh coordinates before
        interpolation.  Use ``1e-3`` when model coordinates are in micrometers and
        the SimNIBS mesh is in millimeters.  Default is ``1.0``.
    coordinate_offset : array-like of shape (3,), optional
        Additive offset applied after scaling, in mesh-coordinate units.

    Notes
    -----
    * Model coordinates and the scalar result have shape ``(*batch, N, C)``.
    * Scalar values retain the units and reference-amplitude normalization of
      the supplied ``NodeData``.  For extracellular stimulation, the
      interpolated field multiplied by its temporal waveform must be in mV.
    * The SimNIBS mesh point-location step remains CPU/NumPy/Cython based inside
      ``PreparedInterpolate3dFEM``; value gathering/blending runs in torch on the
      interpolator device.
    * Gradients may flow to learnable field values when the prepared interpolator
      was constructed with ``learnable_values=True``.  Gradients do not flow
      through the discrete tetrahedron lookup or back to query coordinates.
    """

    def __init__(
        self,
        interpolator,
        *,
        coordinate_scale=1.0,
        coordinate_offset=None,
    ):
        super().__init__()
        self.interpolator = interpolator
        self._init_mesh_coordinate_transform(
            coordinate_scale=coordinate_scale,
            coordinate_offset=coordinate_offset,
        )

    @classmethod
    def from_NodeData(
        cls,
        node_data,
        *,
        out_fill=np.nan,
        th_indices=None,
        coordinate_scale=1.0,
        coordinate_offset=None,
        dtype=None,
        device=None,
        learnable_values=False,
        values_requires_grad=True,
    ):
        """Create a scalar mesh interpolator from a SimNIBS-like ``NodeData``.

        ``NodeData`` is the natural format for scalar nodal fields such as voltage.
        The wrapped prepared interpolator uses tetrahedron containment plus
        barycentric interpolation, matching ``node_data.interpolate_scattered``.
        """
        nr_comp = getattr(node_data, "nr_comp", None)
        if nr_comp is not None and int(nr_comp) != 1:
            raise ValueError(
                "PreComputedInterpolate3DMesh expects scalar NodeData. "
                "For vector E-fields, use EFieldInterpolate3DMesh.from_ElementData."
            )

        interpolator = PreparedInterp3dFEM.from_NodeData(
            node_data,
            out_fill=out_fill,
            th_indices=th_indices,
            squeeze=False,
            dtype=dtype,
            device=device,
            learnable_values=learnable_values,
            values_requires_grad=values_requires_grad,
        )
        return cls(
            interpolator,
            coordinate_scale=coordinate_scale,
            coordinate_offset=coordinate_offset,
        )

    # PEP-8 alias; keep from_NodeData for SimNIBS class-name symmetry.
    from_node_data = from_NodeData

    def _interp(self, x, y, z):
        shape = x.shape
        xyz_q = self._query_xyz_from_model_coords(x, y, z)
        field = self.interpolator(xyz_q, squeeze=False)
        if field.ndim == 2:
            if field.shape[-1] != 1:
                raise RuntimeError(
                    "Expected scalar mesh interpolator output with one component, "
                    f"got shape {tuple(field.shape)}."
                )
            field = field.squeeze(-1)
        return field.reshape(*shape)

    def forward(self, model):
        """Interpolate the scalar field at ``model.x``, ``model.y``, ``model.z``."""
        self.to(device=model.device(), dtype=model.dtype())
        x, y, z = model.x, model.y, model.z
        return self._interp(x, y, z)


# -- aliases --
FEMInterpolate1D = PreComputedInterpolate1D
FEMInterpolate3DRect = PreComputedInterpolate3DRect
FEMInterpolate3DScattered = PreComputedInterpolate3DScattered
FEMInterpolate3DMesh = PreComputedInterpolate3DMesh


class EfieldInterpolate3DRect(torch.nn.Module):
    """Interpolate 3D E-field from rectilinear grid and compute quasipotentials.

    Parameters
    ----------
    x : torch.Tensor
        1D tensor of x-coordinates of the grid points (in μm).
    y : torch.Tensor
        1D tensor of y-coordinates of the grid points (in μm).
    z : torch.Tensor
        1D tensor of z-coordinates of the grid points (in μm).
    efield : torch.Tensor
        4D tensor of shape (Nx, Ny, Nz, 3) containing the electric-field
        vectors at the grid points in V/m.
    **kwargs : additional keyword arguments for PreparedInterp3dRect.

    Notes
    -----
    *  Model coordinates have shape ``(*batch, N, C)``; interpolated vectors
       have one trailing component axis and quasipotentials preserve the model
       coordinate shape.
    *  All operations remain on the same device/dtype as the model.
    *  The module is differentiable w.r.t. the model's coordinates; the
       E-field grid points/values are treated as constants (buffers).
    *  The module is differentiable w.r.t. the efield values, but must be
       specified explicitly with learnable=True as a kwarg.
    *  Calling the module integrates the interpolated E-field along the model
       morphology and returns quasipotentials in mV.
    """

    def __init__(self, x, y, z, efield, **kwargs):
        super().__init__()
        self.interpolator = PreparedInterp3dRect(x=x, y=y, z=z, values=efield, **kwargs)

    def _interp(self, x, y, z):
        shape = x.shape
        xyz_q = torch.stack((x, y, z), dim=-1).reshape(-1, 3)  # (Q, 3)
        efield = self.interpolator(xyz_q).view(*shape, 3)
        return efield

    def forward(self, model):
        self.to(device=model.device(), dtype=model.dtype())
        x, y, z = model.x, model.y, model.z
        G = model.graph
        efield = self._interp(x, y, z)
        return calculate_quasipotentials_batched_coords(G, x, y, z, efield)


class EfieldInterpolate3DScattered(torch.nn.Module):
    """Interpolate 3D E-field from scattered points and compute quasipotentials.

    Parameters
    ----------
    xyz : torch.Tensor
        A tensor of shape (N, 3) containing the coordinates (x, y, z) in μm.
    efield : torch.Tensor
        A tensor of shape (N, 3) containing the electric-field vectors at the
        coordinates in V/m.
    **kwargs : additional keyword arguments for PreparedInterp3dScattered.

    Notes
    -----
    *  Model coordinates have shape ``(*batch, N, C)``; interpolated vectors
       have one trailing component axis and quasipotentials preserve the model
       coordinate shape.
    *  All operations remain on the same device/dtype as the model.
    *  The module is differentiable w.r.t. the model's coordinates; the
       sample points/values are treated as constants (buffers).
    *  The module is differentiable w.r.t. the efield values, but must be
       specified explicitly with learnable=True as a kwarg.
    *  Calling the module integrates the interpolated E-field along the model
       morphology and returns quasipotentials in mV.
    """

    def __init__(
        self,
        xyz,
        efield,
        **kwargs,
    ):
        super().__init__()
        xyz = torch.as_tensor(xyz)
        efield = torch.as_tensor(efield)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3).")
        if efield.shape != xyz.shape:
            raise ValueError("efield must have the same (N, 3) shape as xyz.")

        self.interpolator = PreparedInterp3dScattered(
            points=xyz,
            values=efield,
            **kwargs,
        )

    def _interp(self, x, y, z):
        shape = x.shape
        xyz_q = torch.stack((x, y, z), dim=-1).reshape(-1, 3)  # (Q, 3)
        efield = self.interpolator(xyz_q).view(*shape, 3)
        return efield

    def forward(self, model):
        self.to(device=model.device(), dtype=model.dtype())
        x, y, z = model.x, model.y, model.z
        G = model.graph
        efield = self._interp(x, y, z)
        return calculate_quasipotentials_batched_coords(G, x, y, z, efield)


class EfieldInterpolate3DMesh(_MeshCoordinateTransformMixin, torch.nn.Module):
    """Interpolate a mesh-defined 3-D E-field and compute quasipotentials.

    This is the tetrahedral-mesh analogue of :class:`EfieldInterpolate3DRect`
    and :class:`EfieldInterpolate3DScattered`.  E-field solutions from SimNIBS are
    stored as ``ElementData`` with three components per element, so construction is
    normally via :meth:`from_ElementData`.

    Parameters
    ----------
    interpolator : PreparedInterpolate3dFEM
        Prepared FEM interpolator created from vector ``ElementData``.
    coordinate_scale : float, optional
        Multiplicative conversion from model coordinates to mesh coordinates before
        interpolation.  Use ``1e-3`` when model coordinates are in micrometers and
        the SimNIBS mesh is in millimeters.  Default is ``1.0``.
    coordinate_offset : array-like of shape (3,), optional
        Additive offset applied after scaling, in mesh-coordinate units.

    Notes
    -----
    Model coordinates have shape ``(*batch, N, C)``; interpolated vectors have
    one trailing component axis and quasipotentials preserve the coordinate
    shape.

    The supplied ``ElementData`` values must be electric-field vectors in V/m;
    calling the module returns quasipotentials in mV.

    The default ``from_ElementData(..., method='linear', continuous=False)`` path
    reproduces SimNIBS' tag-wise element-to-node recovery followed by barycentric
    interpolation, which is the appropriate high-quality interpolation path for
    discontinuous E-fields across tissue boundaries.
    """

    def __init__(
        self,
        interpolator,
        *,
        coordinate_scale=1.0,
        coordinate_offset=None,
    ):
        super().__init__()
        self.interpolator = interpolator
        self._init_mesh_coordinate_transform(
            coordinate_scale=coordinate_scale,
            coordinate_offset=coordinate_offset,
        )

    @classmethod
    def from_ElementData(
        cls,
        element_data,
        *,
        out_fill=np.nan,
        method="linear",
        continuous=False,
        th_indices=None,
        coordinate_scale=1.0,
        coordinate_offset=None,
        dtype=None,
        device=None,
        learnable_values=False,
        values_requires_grad=True,
    ):
        """Create an E-field interpolator from SimNIBS-like ``ElementData``.

        Parameters mirror ``ElementData.interpolate_scattered`` where relevant.
        The default ``method='linear', continuous=False`` is selected for E-fields
        because it preserves tag-wise discontinuities by preparing separate
        recovered nodal fields per tissue tag.
        """
        nr_comp = getattr(element_data, "nr_comp", None)
        if nr_comp is not None and int(nr_comp) != 3:
            raise ValueError(
                "EFieldInterp3DMesh expects vector ElementData with exactly "
                f"3 components; got nr_comp={nr_comp}."
            )

        interpolator = PreparedInterp3dFEM.from_ElementData(
            element_data,
            out_fill=out_fill,
            method=method,
            continuous=continuous,
            squeeze=False,
            th_indices=th_indices,
            dtype=dtype,
            device=device,
            learnable_values=learnable_values,
            values_requires_grad=values_requires_grad,
        )
        return cls(
            interpolator,
            coordinate_scale=coordinate_scale,
            coordinate_offset=coordinate_offset,
        )

    # PEP-8 alias; keep from_ElementData for SimNIBS class-name symmetry.
    from_element_data = from_ElementData

    def _interp(self, x, y, z):
        shape = x.shape
        xyz_q = self._query_xyz_from_model_coords(x, y, z)
        efield = self.interpolator(xyz_q, squeeze=False)
        if efield.ndim != 2 or efield.shape[-1] != 3:
            raise RuntimeError(
                "Expected E-field mesh interpolator output with shape (Q, 3), "
                f"got {tuple(efield.shape)}."
            )
        return efield.reshape(*shape, 3)

    def interpolate_efield(self, model):
        """Return interpolated E-field vectors at model coordinates."""
        self.to(device=model.device(), dtype=model.dtype())
        x, y, z = model.x, model.y, model.z
        return self._interp(x, y, z)

    def forward(self, model):
        """Interpolate E-field vectors and convert them to quasipotentials."""
        self.to(device=model.device(), dtype=model.dtype())
        x, y, z = model.x, model.y, model.z
        G = model.graph
        efield = self._interp(x, y, z)
        return calculate_quasipotentials_batched_coords(G, x, y, z, efield)
