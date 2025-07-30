"""Precomputed fields."""

import torch

import glob
from natsort import natsorted
import numpy as np
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from .quasipotentials import calculate_quasipotentials_batched_coords


class PreComputed:
    """Generic abstract base class for PreComputed field potential sources."""

    def __init__(self, data, in_memory=True, is_numpy=False, options=None):
        if in_memory:
            data = np.asarray(data)
            self.data = np.atleast_2d(data) if data.dtype.kind in "biufc" else data
            self.get = getattr(self, "get_in_memory")
        else:
            self.data = data
            self.get = getattr(self, "get_from_disk")
            if is_numpy:
                self.load = np.load
            else:
                self.load = np.loadtxt
            self.options = options or {}

    def init_ve_space(self):
        """Populate and return a list of voltage distributions produced by a
        a 1mA current source at this electrode location at every section
        in every axon in self.axons.
        """
        raise NotImplementedError()

    def get_in_memory(self, gid):
        return np.asarray(self.data[gid])

    def get_from_disk(self, gid):
        loc = self.data.format(gid=gid)
        vals = self.load(loc, **self.options)
        return vals


class PreComputedExact(PreComputed):
    """Use PreComputed data with exact values for every section in every model."""

    name = "pre_computed_exact"

    def init_ve_space(self):
        out = [self.get(axon.gid) for axon in self.axons]
        return out


class PreComputedInterpolate1D(PreComputed):
    """Use PreComputed data sampled along the y-axis at different
    (x, z) locations."""

    name = "pre_computed_interpolate_1d"

    def __init__(
        self,
        data=None,
        y=None,
        method="quadratic",
        fill_value="extrapolate",
        truncate=None,
        in_memory=True,
        is_numpy=False,
        options=None,
    ):
        super(PreComputedInterpolate1D, self).__init__(
            data, in_memory, is_numpy, options
        )
        self.x = np.atleast_2d(np.asarray(y))
        self.method = method
        self.use_point_source = False
        if truncate is not None and truncate > 0.4:
            raise ValueError("Cannot truncate by more than 40%.")
        self.truncate = truncate
        self.fill_value = fill_value
        self.fv = fill_value
        if fill_value == "point_source":
            self.fv = 0
            self.use_point_source = True

    @classmethod
    def from_ascent(cls, ascent_dir, sample, model, sim, contact, **kwargs):
        data = glob.glob(
            f"{ascent_dir}/samples/{sample}/models/{model}/sims/{sim}/fibersets_bases/0/{contact}/*.dat"
        )
        data = natsorted(data)
        data = np.vstack([np.loadtxt(f, skiprows=1) for f in data])
        kwargs["in_memory"] = True
        y = np.loadtxt(
            f"{ascent_dir}/samples/{sample}/models/{model}/sims/{sim}/fibersets/0/0.dat",
            skiprows=1,
        )[:, -1]
        return cls(data=data, y=y, **kwargs)

    def interpolate_batch(self, y_points, indices):
        """Interpolate voltage values along axon."""
        return np.vstack(
            [self.interpolate(yp, idx) for (yp, idx) in tqdm(zip(y_points, indices))]
        )

    def interpolate_batch_indices(self, y_points, indices):
        """Interpolate voltage values along axon."""
        cache = {}
        for i in indices:
            if i not in cache:
                cache[i] = self.interpolate(y_points, i)
        return np.vstack([cache[i] for i in indices])

    def interpolate(self, y_points, idx):
        fem_vec = self.get(idx)
        y_vec = self.x[0]
        return self._interpolate(fem_vec, y_vec, y_points)

    def _interpolate(self, fem_vec, y_vec, y_points):
        if self.truncate:
            t_n = int(len(y_vec) * self.truncate)
            fem_vec = fem_vec[t_n:-t_n]
            y_vec = y_vec[t_n:-t_n]
        if self.use_point_source:
            bounds_error = False
        else:
            bounds_error = None
        interp_func = interp1d(
            y_vec,
            fem_vec,
            kind=self.method,
            assume_sorted=True,
            fill_value=self.fv,
            bounds_error=bounds_error,
        )
        interp = interp_func(y_points)
        if self.use_point_source:
            self.point_source_fill(y_vec, y_points, interp)
        return interp

    def point_source_fill(self, y_vec, y_points, interp):
        peak_y = y_points[np.argmax(interp)]
        peak_v = interp.max()

        # do for y < y_min
        end_v = interp[np.searchsorted(y_points, y_vec.min())]
        end_y = y_points[np.searchsorted(y_points, y_vec.min())]
        pe = peak_v**2 / end_v**2
        x = end_y - peak_y
        d = np.sqrt(-(x**2) / (1 - pe))
        sigma = 1 / (2 * np.pi * d * peak_v)
        y_fill = y_points[y_points <= end_y]
        interp[y_points <= end_y] = 1 / (
            2 * np.pi * sigma * np.sqrt(d**2 + (peak_y - y_fill) ** 2)
        )

        # do for y > y_max
        end_v = interp[np.searchsorted(y_points, y_vec.max(), side="right") - 1]
        end_y = y_points[np.searchsorted(y_points, y_vec.max(), side="right") - 1]
        pe = peak_v**2 / end_v**2
        x = end_y - peak_y
        d = np.sqrt(-(x**2) / (1 - pe))
        sigma = 1 / (2 * np.pi * d * peak_v)
        y_fill = y_points[y_points >= end_y]
        interp[y_points >= end_y] = 1 / (
            2 * np.pi * sigma * np.sqrt(d**2 + (peak_y - y_fill) ** 2)
        )


# -- aliases --
FEMExact = PreComputedExact
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
        G = model.G
        efield = self._interp(x, y, z)
        return calculate_quasipotentials_batched_coords(G, x, y, z, efield)
