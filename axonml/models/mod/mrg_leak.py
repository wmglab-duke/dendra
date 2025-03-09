# McIntyre, Richardson, Grill 2002

from ..mechanisms import *
from ..mechanisms.ops import expit, exprelr


class mrg_leak(Mechanism):
    PARAMETER(gl=0.007, el=-90.0)

    NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.gl * (v - self.el)