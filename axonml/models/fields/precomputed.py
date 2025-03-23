"""Precomputed fields."""

import glob
from natsort import natsorted
import numpy as np
from scipy.interpolate import interp1d
from tqdm.auto import tqdm


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
