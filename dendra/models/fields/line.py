import torch

from ..parametric import SimpleParameterized as P


class Line(P):
    """
    Analytic polyline current source in a homogeneous isotropic conductor.

    The source is represented as a piecewise-linear trajectory in space and is
    interpreted as a *single* line source carrying unit total current that is
    uniformly distributed per unit arc length along the full polyline. The
    potential is computed by summing the closed-form contribution from each
    straight segment, i.e. the standard finite line-source approximation.

    Parameters
    ----------
    xyz : array_like, shape (N, 3)
        Polyline vertices in micrometers (µm). Adjacent vertices define the
        individual straight line segments.
    rhoe : float, optional
        Extracellular resistivity in Ω·cm. Default is 300.0.
    min_distance : float, optional
        Minimum perpendicular distance to the line source in µm. This can be
        used as a geometric regularizer or effective conductor radius to avoid
        the logarithmic singularity that occurs exactly on the source axis.
        Default is 0.0.
    medium : {"infinite", "semi_infinite"}, optional
        ``"infinite"`` uses the standard ``4π`` denominator.
        ``"semi_infinite"`` applies the usual factor-of-two image-source
        scaling via a ``2π`` denominator. This is only an amplitude scaling; it
        is not a full boundary-aware half-space model.
    """

    P.PARAMETER(rhoe=300.0, min_distance=0.0)

    _valid_medium = {"infinite": 4.0, "semi_infinite": 2.0}

    def __init__(self, xyz, rhoe=300.0, min_distance=0.0, medium="infinite"):
        if medium not in self._valid_medium:
            raise ValueError(
                f"Unknown medium '{medium}'. Valid options are {list(self._valid_medium)}."
            )

        super().__init__(rhoe=rhoe, min_distance=min_distance)

        xyz = torch.as_tensor(xyz, dtype=torch.float32)
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape (N, 3).")
        if xyz.shape[0] < 2:
            raise ValueError("xyz must contain at least two vertices.")

        diff = xyz[1:] - xyz[:-1]
        ds = torch.linalg.norm(diff, dim=-1)
        if torch.any(ds <= 0):
            raise ValueError("Adjacent line-source vertices must be distinct.")

        self.medium = medium

        self.register_buffer("xyz", xyz.contiguous())
        self.register_buffer("start", xyz[:-1].contiguous())
        self.register_buffer("end", xyz[1:].contiguous())
        self.register_buffer("diff", diff.contiguous())
        self.register_buffer("ds", ds.contiguous())
        self.register_buffer("direction", diff / ds.unsqueeze(-1))
        self.register_buffer("total_length", ds.sum().reshape(()))

    def fn(self, x, y, z):
        coords = torch.stack([x, y, z], dim=-1)
        original_shape = coords.shape[:-1]
        points = coords.reshape(-1, 3)

        # For a segment parameterized as r(s) = start + s * u, with
        # 0 <= s <= ds, the potential contribution is proportional to
        # ∫ ds / ||point - r(s)||. The closed form is
        #   asinh((ds - h) / r_perp) + asinh(h / r_perp),
        # where h is the projection of (point - start) onto the segment axis
        # and r_perp is the perpendicular distance to that axis.
        rel = points[:, None, :] - self.start[None, :, :]
        h = torch.sum(rel * self.direction[None, :, :], dim=-1)
        r_perp_sq = torch.sum(rel * rel, dim=-1) - h.square()
        r_perp_sq = torch.clamp_min(r_perp_sq, 0.0)
        r_perp = torch.sqrt(r_perp_sq)

        min_distance = torch.clamp(self.min_distance.to(r_perp), min=0.0)
        eps = torch.finfo(r_perp.dtype).eps
        r_perp = torch.clamp(torch.maximum(r_perp, min_distance), min=eps)

        segment_integrals = torch.asinh((self.ds[None, :] - h) / r_perp) + torch.asinh(
            h / r_perp
        )

        scale = (
            1e4
            * self.rhoe
            / (self._valid_medium[self.medium] * torch.pi * self.total_length)
        )
        phi = scale * torch.sum(segment_integrals, dim=-1)
        return phi.reshape(original_shape)

    def forward(self, model):
        self.to(device=model.device(), dtype=model.dtype())
        x, y, z = model.x, model.y, model.z
        return self.fn(x, y, z)


class arbitrary_line(Line):
    """Analytic line source defined by an arbitrary polyline trajectory."""

    def __init__(self, xyz, **kwargs):
        super().__init__(xyz=xyz, **kwargs)


class line3d(Line):
    """Straight 3-D line source between ``start`` and ``end``."""

    def __init__(self, start, end, samples=2, **kwargs):
        samples = int(samples)
        if samples < 2:
            raise ValueError("samples must be >= 2.")

        start = torch.as_tensor(start, dtype=torch.float32)
        end = torch.as_tensor(end, dtype=torch.float32)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("start and end must each have shape (3,).")

        t = torch.linspace(0.0, 1.0, samples, dtype=torch.float32)
        xyz = start[None, :] + (end - start)[None, :] * t[:, None]
        super().__init__(xyz=xyz, **kwargs)


class helix_line(Line):
    """Helical line source parameterized around the x-axis."""

    def __init__(
        self,
        x_start,
        x_end,
        radius,
        orbits,
        phase=0.0,
        samples=100,
        **kwargs,
    ):
        samples = int(samples)
        if samples < 2:
            raise ValueError("samples must be >= 2.")

        theta = torch.linspace(
            0.0, float(orbits) * 2.0 * torch.pi, samples, dtype=torch.float32
        )
        x = torch.linspace(float(x_start), float(x_end), samples, dtype=torch.float32)
        phase = torch.as_tensor(phase, dtype=torch.float32)
        radius = float(radius)

        z = radius * torch.sin(theta + phase)
        y = radius * torch.cos(theta + phase)
        xyz = torch.stack((x, y, z), dim=-1)
        super().__init__(xyz=xyz, **kwargs)


class arc_line(helix_line):
    """Planar arc line source at fixed ``x``."""

    def __init__(self, x, radius, orbit, phase=0.0, samples=100, **kwargs):
        if not (0.0 <= float(orbit) <= 1.0):
            raise ValueError(
                "An arc must have 0 <= orbit <= 1 so it spans at most one full turn."
            )
        super().__init__(
            x_start=x,
            x_end=x,
            radius=radius,
            orbits=orbit,
            phase=phase,
            samples=samples,
            **kwargs,
        )
