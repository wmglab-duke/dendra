from ..core import Unmyelinated
from ..mod import (
    ks,
    kf,
    h,
    nattxs,
    nav1p8,
    nav1p9_slow_inact,
    nakpump,
    kdrTiger,
    kna,
    naoiTiger,
    koiTiger,
    leak,
)
from ..mechanisms import PARAMETER
from ..mechanisms.handler.defaults import c_context


class Tigerholm(Unmyelinated):
    PARAMETER(cm=1e-3, rhoa=35.4)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0,
        dx=10,
        temp=37.0,
        v_init=-55.0,
        method="dufort-frankel",
    ):
        super().__init__(diameters, L, dx, temp, v_init, method)

        with c_context(nai0=11.4, nao0=154.0, ki0=144.9, ko0=5.6):
            self.insert(ks, gbar=0.0069733)
            self.insert(kf, gbar=0.012756)
            self.insert(h, gbar=0.0025377)
            self.insert(nattxs, gbar=0.10664)
            self.insert(nav1p8, gbar=0.24271)
            self.insert(nav1p9_slow_inact, gbar=9.4779e-05)
            self.insert(nakpump, smalla=-0.0047891)
            self.insert(kdrTiger, gbar=0.018002)
            self.insert(kna, gbar=0.00042)
            self.insert(naoiTiger)
            self.insert(koiTiger)
            self.insert(
                leak, gkleak=1.3155237866158132e-05, gnaleak=2.1094052499393e-05
            )
            self.build()
