from dataclasses import dataclass

from axonml.units import um


class Population:
    def __init__(self, base_model, diam=[10.0*um], L=10.0*um, temp=37.0):
        self.base_model = base_model
        self.diam = diam
        self.L = L
        self.temp = temp
        self.n = len(diam)
