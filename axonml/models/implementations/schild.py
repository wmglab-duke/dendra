from ..core import Unmyelinated
from ..mod import (
    leakSchild,
    kd,
    ka,
    can,
    cat,
    kds,
    kca,
    caextscale,
    caintscale,
    capump,
    nacapump,
    nakpumpSchild,
    naf97mean,
    nas97mean,
    naf,
    nas,
)
from ..mechanisms import PARAMETER
from ..mechanisms.handler.defaults import e_context as E, c_context as C

import math


class Schild97(Unmyelinated):
    PARAMETER(cm=1.326291192e-3, rhoa=100.0)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0,
        dx=10,
        temp=37.0,
        v_init=-69.0,
        method="dufort-frankel",
    ):
        super().__init__(diameters, L, dx, temp, v_init, method)
        R = 8314  # molar gas constant
        F = 96500  # Faraday's constant

        ko = 5.4
        ki = 145.0
        ek = ((R * (temp + 273.15)) / F) * math.log(ko / ki)

        nao = 154.0
        nai = 8.9
        ena = ((R * (temp + 273.15)) / F) * math.log(nao / nai)

        self.ion_style("na", 1, 2, 0, 0, 0)
        self.ion_style("k", 1, 2, 0, 0, 0)

        with (
            E(ena=ena, ek=ek),
            C(cao0=2.0, cai0=0.000117, ko0=ko, ki0=ki, nao0=nao, nai0=nai),
        ):
            self.insert(leakSchild, gbna=1.8261e-05, gbca=9.13049e-06)
            self.insert(kd, gbar=0.001956534)
            self.insert(ka, gbar=0.001304356)
            self.insert(can, gbar=0.000521743)
            self.insert(cat, gbar=0.00018261)
            self.insert(kds, gbar=0.000782614)
            self.insert(kca, gbar=0.000913049)
            self.insert(caextscale, lseg=(1e-4) * dx)
            self.insert(caintscale, lseg=(1e-4) * dx)
            self.insert(capump)
            self.insert(nacapump)
            self.insert(nakpumpSchild)
            self.insert(naf97mean, gbar=0.022434928)
            self.insert(nas97mean, gbar=0.022434928)
            self.build()


class Schild94(Unmyelinated):
    PARAMETER(cm=1.326291192e-3, rhoa=100.0)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0,
        dx=10,
        temp=37.0,
        v_init=-48.0,
        method="dufort-frankel",
    ):
        super().__init__(diameters, L, dx, temp, v_init, method)
        R = 8314  # molar gas constant
        F = 96500  # Faraday's constant

        ko = 5.4
        ki = 145.0
        ek = ((R * (temp + 273.15)) / F) * math.log(ko / ki)

        nao = 154.0
        nai = 8.9
        ena = ((R * (temp + 273.15)) / F) * math.log(nao / nai)

        self.ion_style("na", 1, 2, 0, 0, 0)
        self.ion_style("k", 1, 2, 0, 0, 0)

        with (
            E(ena=ena, ek=ek),
            C(cao0=2.0, cai0=0.000117, ko0=ko, ki0=ki, nao0=nao, nai0=nai),
        ):
            self.insert(leakSchild)
            self.insert(kd)
            self.insert(ka)
            self.insert(can)
            self.insert(cat)
            self.insert(kds)
            self.insert(kca)
            self.insert(caextscale, lseg=(1e-4) * dx)
            self.insert(caintscale, lseg=(1e-4) * dx)
            self.insert(capump)
            self.insert(nacapump)
            self.insert(nakpumpSchild)
            self.insert(naf)
            self.insert(nas)
            self.build()
