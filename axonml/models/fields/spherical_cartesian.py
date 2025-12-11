import torch


def spherical_to_cartesian(spherical_vecs: torch.Tensor) -> torch.Tensor:
    """
    Calculates cartesian coordinates from spherical coordinates using PyTorch.
    Assumes degrees for input angles.

    Parameters
    ----------
    spherical_vecs : torch.Tensor (..., 3)
        Spherical coordinates in degrees in the format [phi, theta, r].
        phi: azimuthal angle (0 to 360)
        theta: polar/inclination angle (0 to 180)
        r: radius/magnitude

    Returns
    -------
    torch.Tensor (..., 3)
        Cartesian coordinates in the format [x, y, z]
    """
    # Ensure input is a tensor
    if not isinstance(spherical_vecs, torch.Tensor):
        spherical_vecs = torch.tensor(spherical_vecs)

    # Decompose the spherical coordinates
    phi_deg = spherical_vecs[..., 0]
    theta_deg = spherical_vecs[..., 1]
    r = spherical_vecs[..., 2]

    # Convert angles from degrees to radians for PyTorch's trig functions
    phi_rad = torch.deg2rad(phi_deg)
    theta_rad = torch.deg2rad(theta_deg)

    # Calculate Cartesian coordinates
    x = r * torch.cos(phi_rad) * torch.sin(theta_rad)
    y = r * torch.sin(phi_rad) * torch.sin(theta_rad)
    z = r * torch.cos(theta_rad)

    # Stack the results into a single tensor
    return torch.stack([x, y, z], dim=-1)
