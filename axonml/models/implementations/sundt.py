from ..core import Unmyelinated
from ..mod import kdr, pas, nahh
from ..mechanisms import PARAMETER
from ..mechanisms.handler.defaults import e_context


class Sundt(Unmyelinated):
    PARAMETER(cm=1e-3, rhoa=100.0)

    def __init__(
        self, n_ax=1, L=5.0, dx=10, temp=37.0, v_init=-65.0, method="rk1"
    ):
        super().__init__(n_ax, L, dx, temp, v_init, method)

        with e_context(ek=-90.0):
            self.insert(kdr, gkbar=0.04)
            self.insert(nahh, gnabar=0.04)
            self.insert(pas, g=0.0001, e=-65.0)
            self.build()
