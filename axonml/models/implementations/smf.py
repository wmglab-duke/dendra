from ..core import Myelinated
from ..mod import axnode_myel

class SMF(Myelinated):
    def __init__(self, temp=37.0, v_init=-80.0):
        super().__init__(temp, v_init)
        self.insert(
            axnode_myel, ic={
            'm': 0.0732093,
            'h': 0.62069505,
            'p': 0.20260409,
            's': 0.04302994
        })