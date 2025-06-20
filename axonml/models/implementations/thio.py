from ..core import Unmyelinated
from ..mod import (
    nav7,
    newnav8,
    nav9,
    bk,
    cav12,
    cav22,
    caextscale,
    caintscale,
    km,
    hcn,
    kv21,
    ka34,
    ka14,
    sk,
    nacx,
    naoi,
    koi,
    nakpumpSchild,
    extrapump,
)
from ..mechanisms import PARAMETER, e_context as E, c_context as C

from axonml.units import mm
from axonml.models.integrators import bwd_euler_ub

import math


def pre_init(model):
    model.mech.extrapump.pumpina.zero_()
    model.mech.extrapump.pumpik.zero_()
    model.mech.extrapump.pumpica.zero_()


def balance(model):
    model.mech.extrapump.pumpina.data.copy_(
        -model.mech.na_ion.ina.flatten()[0]
    )
    model.mech.extrapump.pumpik.data.copy_(
        -model.mech.k_ion.ik.flatten()[0]
    )
    model.mech.extrapump.pumpica.data.copy_(
        -model.mech.ca_ion.ica.flatten()[0]
    )


class ThioAutonomic2025(Unmyelinated):
    PARAMETER(cm=1.326291192, rhoa=23.117539, celsius=37.0)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0*mm,
        dx=10,
        celsius=37.0,
        v_init=-58.5,
        integrator=None,
    ):
        if integrator is None:
            integrator = bwd_euler_ub()
        super().__init__(diameters, L, dx, celsius, v_init, integrator)
        R = 8314  # molar gas constant
        F = 96485.3329  # Faraday's constant

        ko = 5.4
        ki = 145.0
        ek = ((R * (celsius + 273.15)) / F) * math.log(ko / ki)
        ki_real = 144.9
        ko_real = 5.6

        nao = 154.0
        nai = 8.9
        ena = ((R * (celsius + 273.15)) / F) * math.log(nao / nai)
        nai_real = 11.4

        self.ion_style("na", 3, 2, 1, 1, 0)
        self.ion_style("k", 3, 2, 1, 1, 0)

        self.register_pre_initialize_hook(pre_init)
        self.register_post_initialize_hook(balance)

        with (
            E(ena=ena, ek=ek),
            C(cao0=2.0, cai0=0.000117, ko0=ko_real, ki0=ki_real, nao0=nao, nai0=nai_real),
        ):
            self.insert(nav7, gbar=0.036813)
            self.insert(newnav8, gbar=0.075747)
            self.insert(nav9, gbar=0.000376)
            self.insert(bk, gbar=0.000156)
            self.insert(cav12, gbar=0.000004)
            self.insert(cav22, gbar=0.009546)
            self.insert(km, gbar=0.002864)
            self.insert(caextscale, lseg=(1e-4) * dx, fhspace=.03)
            self.insert(caintscale, lseg=(1e-4) * dx)
            self.insert(hcn, gbar=0.002789)
            self.insert(kv21, gbar=0.005337)
            self.insert(ka34, gbar=0.000289)
            self.insert(ka14, gbar=0.000024)
            self.insert(sk, gbar=0.000006)
            self.insert(nacx, gbar=0.000210)
            self.insert(nakpumpSchild, INaKmax22=0.056316)
            self.insert(naoi)
            self.insert(koi)
            self.insert(extrapump)


class ThioCutaneous2025(Unmyelinated):
    PARAMETER(cm=1.326291192, rhoa=27.513088, celsius=37.0)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0*mm,
        dx=10,
        celsius=37.0,
        v_init=-58.5,
        integrator=None,
    ):
        if integrator is None:
            integrator = bwd_euler_ub()
        super().__init__(diameters, L, dx, celsius, v_init, integrator)
        R = 8314  # molar gas constant
        F = 96485.3329  # Faraday's constant

        ko = 5.4
        ki = 145.0
        ek = ((R * (celsius + 273.15)) / F) * math.log(ko / ki)
        ki_real = 144.9
        ko_real = 5.6

        nao = 154.0
        nai = 8.9
        ena = ((R * (celsius + 273.15)) / F) * math.log(nao / nai)
        nai_real = 11.4

        self.ion_style("na", 3, 2, 1, 1, 0)
        self.ion_style("k", 3, 2, 1, 1, 0)

        self.register_pre_initialize_hook(pre_init)
        self.register_post_initialize_hook(balance)

        with (
            E(ena=ena, ek=ek),
            C(cao0=2.0, cai0=0.000117, ko0=ko_real, ki0=ki_real, nao0=nao, nai0=nai_real),
        ):
            self.insert(nav7, gbar=0.035663)
            self.insert(newnav8, gbar=0.115643)
            self.insert(nav9, gbar=0.000504)
            self.insert(bk, gbar=0.002016)
            self.insert(cav12, gbar=0.000188)
            self.insert(cav22, gbar=0.000361)
            self.insert(km, gbar=0.000003)
            self.insert(caextscale, lseg=(1e-4) * dx, fhspace=.03)
            self.insert(caintscale, lseg=(1e-4) * dx)
            self.insert(hcn, gbar=0.000106)
            self.insert(kv21, gbar=0.327196)
            self.insert(ka34, gbar=0.001786)
            self.insert(ka14, gbar=0.000044)
            self.insert(sk, gbar=0.000755)
            self.insert(nacx, gbar=0.009242)
            self.insert(nakpumpSchild, INaKmax22=0.000456)
            self.insert(naoi)
            self.insert(koi)
            self.insert(extrapump)
