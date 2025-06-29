from ..core import Unmyelinated
from ..mod import rattay_aberham
from ..mechanisms import PARAMETER, equilibria
from axonml.units import mm
from axonml.models.integrators import bwd_euler_ub


class Rattay1993(Unmyelinated):
    PARAMETER(inherit=Unmyelinated, rhoa=100.0)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0*mm,
        dx=10,
        temp=37.0,
        v_init=-70.0,
        integrator=None,
    ):
        if integrator is None:
            integrator = bwd_euler_ub()
        super().__init__(diameters, L, dx, temp, v_init, integrator)

        with equilibria(ena=45.0, ek=-82.0):
            self.insert(rattay_aberham)
