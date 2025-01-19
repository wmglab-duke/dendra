from ..core import Myelinated
from ..mod import sweeney
from ..mechanisms import PARAMETER


class Sweeney(Myelinated):
    PARAMETER(inherit=Myelinated, node_l=1.5, membrane={"cm": 2.5e-3, "rhoa": 54.7})

    def __init__(
        self,
        diameters=[10.0],
        n_node=101,
        temp=37.0,
        v_init=-80.0,
        method="dufort-frankel",
        pade=True,
    ):
        super().__init__(diameters, n_node, temp, v_init, method, pade)
        self.insert(sweeney)
        self.build()
