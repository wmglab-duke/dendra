from ..mechanisms import *
from ..mechanisms.ops import *


class ki(State):
    PARAMETER(FARADAY=96520)
    DERIVATIVE("ki' = -ik*4/FARADAY/diam*(1e4)")


class ko(State):
    PARAMETER(FARADAY=96520, theta=0.03, D=0.1e-6, koinf=5.6)
    DERIVATIVE("ko' = (ik/FARADAY - 0.1*D*(ko-koinf)) / theta*(1e4)")


class koi(Mechanism):
    STATE(ki, ko)
    USEION("k", read=["ik"], write=["ko", "ki"])
