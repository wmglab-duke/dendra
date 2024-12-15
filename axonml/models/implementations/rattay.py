from ..core import Unmyelinated
from ..mod import rattay_aberham
from ..mechanisms import PARAMETER
from ..mechanisms.handler.defaults import e_context


class Rattay(Unmyelinated):
    PARAMETER(cm=1e-3, rhoa=100.0)

    def __init__(self, n_ax=1, n_node=1001, dx=10, temp=37.0, v_init=-70.0, method="euler"):
        super().__init__(n_ax, n_node, dx, temp, v_init, method)

        with e_context(ena=45.0, ek=-82.0):
            self.insert(rattay_aberham)
            self.build()
