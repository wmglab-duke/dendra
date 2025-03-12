from ..core import Unmyelinated
from ..mod import rattay_aberham
from ..mechanisms import PARAMETER, e_context


class Rattay(Unmyelinated):
    PARAMETER(inherit=Unmyelinated, rhoa=100.0)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0,
        dx=10,
        temp=37.0,
        v_init=-70.0,
        method="dufort-frankel",
    ):
        super().__init__(diameters, L, dx, temp, v_init, method)

        with e_context(ena=45.0, ek=-82.0):
            self.insert(rattay_aberham)
