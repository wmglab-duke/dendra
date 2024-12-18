from ..core import Myelinated
from ..mod import axnode_myel


ic = {"m": 0.0732093, "h": 0.62069505, "p": 0.20260409, "s": 0.04302994}


class SMF(Myelinated):
    def __init__(
        self, diameters=[8.0], n_node=101, temp=37.0, v_init=-80.0, method="rk1"
    ):
        super().__init__(diameters, n_node, temp, v_init, method)
        self.insert(axnode_myel, ic=ic)
        self.build()
