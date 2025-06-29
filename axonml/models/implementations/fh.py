from ..core import Myelinated, Unmyelinated
from ..mod import fh
from ..mechanisms import PARAMETER, concentrations as C

from axonml.units import mm
from axonml.models.integrators import euler


class FHM(Myelinated):
    PARAMETER(inherit=Myelinated, node_l=2.5, membrane={"cm": 2.0, "rhoa": 110.0})

    def __init__(
        self,
        diameters=[18.0],
        n_node=101,
        temp=20.0,
        v_init=-70.0,
        integrator=euler(),
    ):
        super().__init__(diameters, n_node, temp, v_init, integrator)
        with C(nai0=13.74, nao0=114.5, ki0=120.0, ko0=2.5):
            self.insert(fh)


SENN = FHM


class FHUM(Unmyelinated):
    PARAMETER(cm=2.0, rhoa=110.0)

    def __init__(
        self,
        diameters=[2.0],
        L=5.0*mm,
        dx=25.0,
        temp=20.0,
        v_init=-70.0,
        integrator=euler(),
    ):
        super().__init__(diameters, L, dx, temp, v_init, integrator)
        with C(nai0=13.74, nao0=114.5, ki0=120.0, ko0=2.5):
            self.insert(fh)
