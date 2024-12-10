from ..core import Unmyelinated
from ..mod import kdr, pas, nahh
from ..mechanisms import PARAMETER


class Sundt(Unmyelinated):
    PARAMETER({"cm": 1e-3, "rhoa": 100.0})

    def __init__(self, dx=10, temp=37.0, v_init=-65.0):
        super().__init__(dx, temp, v_init)
        self.insert(kdr, gkbar=0.04, ek=-90.0)
        self.insert(nahh, gnabar=0.04)
        self.insert(pas, g=0.0001, e=-65.0)
        self.finalize()
