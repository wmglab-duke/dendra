# Extracellular calcium ion accumulation

from ..mechanisms import *
from ..mechanisms.ops import *


class cao(State):
    PARAMETER(
        lseg=1.0,
        VolSchild=1.41372e-08,
        Vol_periSchild=1.46136e-09,
        txfer=4511.0,
        FARADAY=96500,
        cabath=2,
        fhspace=1,
    )

    BUFFERS("SA", "Vol_peri")
    DERIVATIVE("cao' = ica*SA/(2*Vol_peri*FARADAY) + (cabath - cao)/txfer")

    def initial(self, v):
        self.SA = torch.pi * (1e-4) * self.diam * self.lseg
        Vol = torch.pi * ((1e-4) * (self.diam / 2)) ** 2 * self.lseg
        self.Vol_peri = (
            torch.pi * ((1e-4) * ((self.diam + self.fhspace) / 2)) ** 2 * self.lseg
        ) - Vol


class caextscale(Mechanism):
    STATE(cao)
    USEION("ca", read=["ica"], write=["cao"])
