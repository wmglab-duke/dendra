from ..core import Myelinated, Unmyelinated
from ..mod import fh
from ..mechanisms import PARAMETER, c_context as C

# TODO: Fix (does not generate AP)


class FHM(Myelinated):
    PARAMETER(inherit=Myelinated, node_l=2.5, membrane={"cm": 2e-3, "rhoa": 110.0})

    def __init__(
        self,
        diameters=[18.0],
        n_node=101,
        temp=37.0,
        v_init=-70.0,
        method="dufort-frankel",
        pade=None,
    ):
        super().__init__(diameters, n_node, temp, v_init, method, pade)
        with C(nai0=13.74, nao0=114.5, ki0=120.0, ko0=2.5):
            self.insert(fh)
            self.build()


SENN = FHM


class FHUM(Unmyelinated):
    PARAMETER(
        inherit=Unmyelinated,
        cm=2e-3,
    )

    def __init__(
        self,
        diameters=[2.0],
        L=5.0,
        dx=10,
        temp=37.0,
        v_init=-70.0,
        method="dufort-frankel",
        pade=None,
    ):
        super().__init__(diameters, L, dx, temp, v_init, method, pade)
        with C(nai0=13.74, nao0=114.5, ki0=120.0, ko0=2.5):
            self.insert(fh)
            self.build()
