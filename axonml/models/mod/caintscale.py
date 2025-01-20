# Intracellular calcium ion accumulation

import math

from ..mechanisms import *
from ..mechanisms.ops import *


class oc_cai(State):
    coupled = True

    STATE("oc", "cai")

    PARAMETER(lseg=1e-3, ku=100, kr=0.238, nb=4.0, Bi=0.001, FARADAY=96500)

    BUFFERS("SA", "Vol")

    DERIVATIVE(
        [
            "oc' = ku * cai * (1-oc) - kr * oc",
            "cai' = -ica * (SA) / Vol / (2*FARADAY) - (nb * Bi * (ku * cai * (1 - oc) - kr * oc))",
        ]
    )

    def initial(self, v):
        self.SA = math.pi * (1e-4) * self.diam * self.lseg
        self.Vol = math.pi * ((1e-4) * (self.diam / 2)) ** 2 * self.lseg


class caintscale(Mechanism):
    STATE(oc_cai)
    USEION("ca", read=["ica"], write=["cai"])

    INITIAL(oc=0.05)
