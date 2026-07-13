from typing import List, Union

import torch

from ..parametric import SimpleParameterized as P
from .quasipotentials import calculate_quasipotentials_batched_coords
from .spherical_cartesian import spherical_to_cartesian


class Point(P):
    P.PARAMETER(x=0.0, y=0.0, z=0.0)

    def fn(self, x, y, z):
        raise NotImplementedError()

    def forward(self, model):
        self.to(model.device())
        x, y, z = model.x, model.y, model.z
        return self.fn(x, y, z)


class isotropic_point(Point):
    """
    Isotropic point source for electric potential calculation.

    Calculates the electric potential in a medium with uniform (isotropic) resistivity,
    where the resistivity is the same in all directions.

    Parameters
    ----------
    x : float, optional
        X-coordinate of the point source in μm. Default is 0.0.
    y : float, optional
        Y-coordinate of the point source in μm. Default is 0.0.
    z : float, optional
        Z-coordinate of the point source in μm. Default is 0.0.
    rhoe : float, optional
        Extracellular resistivity in Ω·cm. Default is 300.0.

    Notes
    -----
    The potential is calculated using the standard point source equation:

    .. math::
        \\Phi(x,y,z) = \\frac{\\rho_e}{4\\pi r}

    where :math:`r = \\sqrt{(x-x_0)^2 + (y-y_0)^2 + (z-z_0)^2}
    \\cdot 10^{-4}` is the source distance in cm.

    The result has units of Ω and is represented numerically as a lead field in
    mV/mA, because Ω × mA = mV.  Equivalently, it is the potential in mV
    produced by a 1 mA point source.
    """

    Point.PARAMETER(x=0.0, y=0.0, z=0.0, rhoe=300.0)

    def fn(self, x, y, z):
        r = torch.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2) * 1e-4
        return self.rhoe / (4 * torch.pi * r)


class anisotropic_point(Point):
    """
    Anisotropic point source for electric potential calculation.

    Calculates the electric potential in a medium with anisotropic resistivity,
    where the resistivity values can be different along the x, y, and z axes.

    Parameters
    ----------
    x : float, optional
        X-coordinate of the point source in μm. Default is 0.0.
    y : float, optional
        Y-coordinate of the point source in μm. Default is 0.0.
    z : float, optional
        Z-coordinate of the point source in μm. Default is 0.0.
    rhox : float, optional
        Resistivity in the x-direction in Ω·cm. Default is 300.0.
    rhoy : float, optional
        Resistivity in the y-direction in Ω·cm. Default is 300.0.
    rhoz : float, optional
        Resistivity in the z-direction in Ω·cm. Default is 300.0.

    Notes
    -----
    The potential is calculated using the anisotropic medium equation:

    .. math::
        \\Phi(x,y,z) = \\frac{1}{4\\pi
        \\sqrt{\\frac{(x-x_0)^2}{\\rho_y \\rho_z}
        + \\frac{(y-y_0)^2}{\\rho_x \\rho_z}
        + \\frac{(z-z_0)^2}{\\rho_x \\rho_y}}}

    where all coordinate differences are converted from μm to cm before the
    expression is evaluated.

    The result has units of Ω and is represented numerically as a lead field in
    mV/mA, because Ω × mA = mV.  Equivalently, it is the potential in mV
    produced by a 1 mA point source.
    """

    Point.PARAMETER(x=0.0, y=0.0, z=0.0, rhox=300.0, rhoy=300.0, rhoz=300.0)

    def fn(self, x, y, z):
        # Convert distances from μm to cm (1e-4)
        dx = (x - self.x) * 1e-4
        dy = (y - self.y) * 1e-4
        dz = (z - self.z) * 1e-4

        # Calculate anisotropic distance term
        r_aniso = torch.sqrt(
            (dx**2) / (self.rhoy * self.rhoz)
            + (dy**2) / (self.rhox * self.rhoz)
            + (dz**2) / (self.rhox * self.rhoy)
        )

        # Lead field in Ω, represented numerically as mV/mA.
        return 1 / (4 * torch.pi * r_aniso)


class parametric_efield(torch.nn.Module):
    """
    Generate quasipotentials for a grid of uniform E-field directions.

    The public :meth:`forward` path treats ``model.x``, ``model.y``, and
    ``model.z`` as coordinates in µm, accepts an E-field magnitude in V/m,
    and returns extracellular quasipotentials in mV. The calculation is fully
    vectorized over the sampled azimuthal and polar directions.

    Parameters
    ----------
    n_azimuthal : int
        The number of azimuthal angles (phi) to sample (e.g., around the z-axis).
    n_polar : int
        The number of polar/inclination angles (theta) to sample (from the z-axis).
    relative_mag_change_per_mm : Union[float, List[float], torch.Tensor]
        Percentage change in E-field magnitude per mm along the
        somatodendritic axis (z-axis). For example, ``200.0`` means 200% per
        mm. Can be a single value or a list of values. A positive value means
        the E-field magnitude increases from the soma towards negative z.
        This parameter is used by the internal spatial-gradient integration
        path; the public :meth:`forward` path generates a uniform field.

    Notes
    -----
    ``e_field_strength_Vm`` is the physical E-field magnitude in V/m, not a
    Dendra conversion scalar. The returned value already has units of mV, so
    when it is used as the spatial component of extracellular stimulation it
    should normally be paired with a dimensionless relative waveform.
    """

    def __init__(
        self,
        n_azimuthal: int,
        n_polar: int,
        relative_mag_change_per_mm: Union[float, List[float], torch.Tensor] = 0.0,
    ):
        super().__init__()
        self.n_phi = n_azimuthal
        self.n_theta = n_polar

        if isinstance(relative_mag_change_per_mm, (int, float)):
            mag_changes = [float(relative_mag_change_per_mm)]
        else:
            mag_changes = list(relative_mag_change_per_mm)

        self.register_buffer(
            "mag_changes", torch.tensor(mag_changes, dtype=torch.float32)
        )
        self.n_mag_changes = len(self.mag_changes)

        phi_vals = torch.linspace(0, 360, self.n_phi + 1)[:-1]
        theta_vals = torch.linspace(0, 180, self.n_theta)

        grid_phi, grid_theta = torch.meshgrid(phi_vals, theta_vals, indexing="ij")

        # Shape: (n_phi, n_theta, 2)
        spherical_directions = torch.stack([grid_phi, grid_theta], dim=-1)
        self.register_buffer("spherical_directions", spherical_directions)

        angles = spherical_directions.reshape(-1, 2)
        phi = angles[:, 0] * (torch.pi / 180.0)
        theta = angles[:, 1] * (torch.pi / 180.0)

        self.register_buffer("phi", phi)
        self.register_buffer("theta", theta)

    def __forward(
        self, model: object, e_field_strength_Vm: float = 1.0
    ) -> torch.Tensor:
        """
        Calculates the E-field vectors for all compartments and conditions.

        Parameters
        ----------
        model : object
            Model whose ``x``, ``y``, and ``z`` tensors contain compartment
            coordinates in µm.
        e_field_strength_Vm : float, default 1.0
            Reference E-field magnitude in V/m.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            E-field vectors in V/m and integrated quasipotentials in mV.
        """
        self.to(device=model.device(), dtype=model.dtype())
        x, y, z = model.x, model.y, model.z

        z_coords = z[0]
        n_compartments = len(z_coords)

        # Reshape tensors for broadcasting to the final shape:
        # (n_phi, n_theta, n_mag_changes, n_compartments)

        # mag_changes: (1, 1, n_mag_changes, 1)
        mag_change_factor = (self.mag_changes / 100.0).view(1, 1, self.n_mag_changes, 1)

        # z_coords: (1, 1, 1, n_compartments)
        z_reshaped = (
            z_coords.view(1, 1, 1, n_compartments) / 1000.0
        )  # Convert from µm to mm

        # Calculate magnitude at each compartment for each condition
        magnitude = 1 - z_reshaped * mag_change_factor

        # Clamp magnitude to be non-negative, as in the original code
        magnitude = torch.clamp(magnitude, min=0).unsqueeze(
            -1
        )  # Shape: (n_phi, n_theta, n_mag_changes, n_compartments, 1)

        # --- Assemble the final spherical vectors ---
        # We need a tensor of shape (n_phi, n_theta, n_mag_changes, n_compartments, 3)
        # to hold [phi, theta, r] for every case.

        # Base phi and theta directions: (n_phi, n_theta, 1, 1, 2)
        base_directions = self.spherical_directions.view(
            self.n_phi, self.n_theta, 1, 1, 2
        )

        # Create the final spherical tensor by broadcasting
        # This is more efficient than creating a large empty tensor and filling it.
        final_spherical_vecs = base_directions.expand(
            self.n_phi, self.n_theta, self.n_mag_changes, n_compartments, 2
        )
        # Now we have [phi, theta]. We need to add magnitude (r).
        final_spherical_vecs = torch.cat(
            [
                final_spherical_vecs,
                magnitude.expand(
                    self.n_phi, self.n_theta, self.n_mag_changes, n_compartments, 1
                ),
            ],
            dim=-1,
        )

        e_fields_normalized = spherical_to_cartesian(final_spherical_vecs)

        # final_spherical_vecs now has shape (n_phi, n_theta, n_mag_changes, n_compartments, 3)
        # reshape it to (n_phi * n_theta * n_mag_changes, n_compartments, 3)
        e_fields_physical_Vm = e_fields_normalized * e_field_strength_Vm

        # 3. Reshape for batching
        e_fields_batched = e_fields_physical_Vm.view(-1, n_compartments, 3)

        # --- (The rest is the same) ---
        num_conditions = e_fields_batched.shape[0]
        x_batch = x.expand(num_conditions, -1)
        y_batch = y.expand(num_conditions, -1)
        z_batch = z.expand(num_conditions, -1)

        quasi_potentials = calculate_quasipotentials_batched_coords(
            G=model.graph,
            x_batch=x_batch,
            y_batch=y_batch,
            z_batch=z_batch,
            e_fields_batch=e_fields_batched,
        )

        return e_fields_batched, quasi_potentials.contiguous()

    def forward(self, model: object, e_field_strength_Vm: float = 1.0) -> torch.Tensor:
        """Return uniform-field quasipotentials for every sampled direction.

        Parameters
        ----------
        model : object
            Model whose ``x``, ``y``, and ``z`` tensors contain compartment
            coordinates in µm.
        e_field_strength_Vm : float, default 1.0
            Uniform E-field magnitude in V/m.

        Returns
        -------
        torch.Tensor
            Quasipotentials in mV with shape
            ``(n_azimuthal * n_polar, n_compartments)`` (plus any compatible
            model batch dimensions).
        """
        self.to(device=model.device(), dtype=model.dtype())
        x, y, z = model.x, model.y, model.z
        x = x / 1e6  # Convert from µm to m
        y = y / 1e6  # Convert from µm to m
        z = z / 1e6  # Convert from µm to m

        phi = self.phi.view(-1, 1)
        theta = self.theta.view(-1, 1)

        ve = (
            -e_field_strength_Vm
            * (
                x * torch.sin(theta) * torch.cos(phi)
                + y * torch.sin(theta) * torch.sin(phi)
                + z * torch.cos(theta)
            )
            * 1000.0
        )  # Convert to mV

        return ve
