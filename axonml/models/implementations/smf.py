from ..core import Myelinated
from ..mod import axnode_myel


ic = {"m": 0.0732093, "h": 0.62069505, "p": 0.20260409, "s": 0.04302994}


class SMF(Myelinated):
    def __init__(self, n_ax, n_node, temp=37.0, v_init=-80.0):
        super().__init__(n_ax, n_node, temp, v_init)
        self.insert(axnode_myel, ic=ic)
        self.build()
