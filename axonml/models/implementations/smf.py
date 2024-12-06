from ..core import Myelinated
from ..mod import Axnode_Myel

class SMF(Myelinated):
    def __init__(self, temp=37.0, v_init=-80.0):
        super().__init__(temp, v_init)
        self.insert(Axnode_Myel)