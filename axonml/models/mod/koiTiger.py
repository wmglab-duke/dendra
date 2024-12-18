from ..mechanisms import *
from ..mechanisms.ops import *


class ki(State):
    PARAMETER(FARADAY=96520)
    DERIVATIVE("ki' = -ik*4/FARADAY/diam*(1e4)")


class nao(State):
    PARAMETER(FARADAY=96520, theta=29.0e-3, D=0.1e-6, koinf=5.6)
    DERIVATIVE("ko' = (ik/FARADAY - 0.1*D*(ko-koinf)) / theta*(1e4)")


class koiTiger(Mechanism):
    USEION("k", read=["ik"], write=["ko", "ki"])
