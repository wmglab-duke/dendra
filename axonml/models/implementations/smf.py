from ..core import Myelinated
from ..mod import axnode_myel
from ..mechanisms.declarations import PARAMETER


ic = {"m": 0.0732093, "h": 0.62069505, "p": 0.20260409, "s": 0.04302994}


class SMF(Myelinated):
    PARAMETER(
        node_l=1.0,
        axon_d={
            "axond1": 0.0187623,
            "axond2": 4.787487e-01,
            "axond3": 1.203613e-01,
        },
        node_d={
            "noded1": 6.303781e-03,
            "noded2": 2.070544e-01,
            "noded3": 5.339006e-01,
        },
        delta_x={
            "deltax1": -8.215284e00,
            "deltax2": 2.724201e02,
            "deltax3": -7.802411e02,
        },
        membrane={
            "cm": 10e-3,
            "rhoa": 70.0,  # ohm-cm
        },
    )

    def __init__(
        self,
        diameters=[8.0],
        n_node=101,
        temp=37.0,
        v_init=-80.0,
        method="rk1",
        pade=None,
    ):
        super().__init__(diameters, n_node, temp, v_init, method, pade)
        self.insert(axnode_myel, ic=ic)
        self.build()
