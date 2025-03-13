import torch

from ..parametric import Parameterized
from ..declarations import PARAMETER


class Point(torch.jit.ScriptModule, Parameterized):

    PARAMETER(x=0.0, y=0.0, z=0.0)

    def __init__(self, **kwargs):
        super().__init__()
        self.instantiate_parameters(**kwargs)

    def fn(self, x, y, z):
        raise NotImplementedError
    
    def forward(self, model):
        x, y, z = model.x(), model.y, model.z
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
        Extracellular resistivity in Ω·cm. Default is 500.0.
        
    Notes
    -----
    The potential is calculated using the standard point source equation:
    
    .. math::
        V(x,y,z) = \\frac{1000 \\cdot \\rho_e}{4\\pi \\cdot r}
        
    where :math:`r = \\sqrt{(x-x_0)^2 + (y-y_0)^2 + (z-z_0)^2} \\cdot 10^{-4}`
    
    The result is in mV with an assumed unit current source, and the distance
    is converted from μm to cm for calculation.
    """
    PARAMETER(x=0.0, y=0.0, z=0.0, rhoe=500.0)

    def fn(self, x, y, z):
        r = torch.sqrt((x-self.x)**2 + (y-self.y)**2 + (z-self.z)**2) * 1e-4
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
        Resistivity in the x-direction in Ω·cm. Default is 500.0.
    rhoy : float, optional
        Resistivity in the y-direction in Ω·cm. Default is 500.0.
    rhoz : float, optional
        Resistivity in the z-direction in Ω·cm. Default is 500.0.
        
    Notes
    -----
    The potential is calculated using the anisotropic medium equation:

    .. math::
        V(x,y,z) = \\frac{1000}{4\\pi \\cdot \\sqrt{\\frac{(x-x_0)^2}{\\rho_x} + \\frac{(y-y_0)^2}{\\rho_y} + \\frac{(z-z_0)^2}{\\rho_z}}}

    where distances are converted from μm to cm (× 10⁻⁴) for calculation.

    The result is in mV with an assumed unit current source.
    """
    PARAMETER(x=0.0, y=0.0, z=0.0, rhox=500.0, rhoy=500.0, rhoz=500.0)

    def fn(self, x, y, z):
        # Convert distances from μm to cm (1e-4)
        dx = (x - self.x) * 1e-4
        dy = (y - self.y) * 1e-4
        dz = (z - self.z) * 1e-4
        
        # Calculate anisotropic distance term
        r_aniso = torch.sqrt((dx**2)/self.rhox + (dy**2)/self.rhoy + (dz**2)/self.rhoz)
        
        # Calculate potential in mV
        return 1 / (4 * torch.pi * r_aniso)

