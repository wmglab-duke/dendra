from ..mechanisms import *
from ..mechanisms.ops import *


class nai(State):
    PARAMETER(FARADAY=96520)
    DERIVATIVE("nai' = -ina*4/FARADAY/diam*(1e4)")


class nao(State):
    PARAMETER(FARADAY=96520, theta=30e-3, D=0.1e-6, naoinf=154.0)
    DERIVATIVE("nao' = (ina/FARADAY - 0.1*D*(nao-naoinf)) / theta*(1e4)")


class naoi(Mechanism):
    STATE(nai, nao)
    USEION("na", read=["ina"], write=["nao", "nai"])
