# Borg-Graham type KDR channel; Borg-Graham 1987

from ..mechanisms import (
    Mechanism, State, PARAMETER, STATE, INITIAL, USEQ10
)
from ..mechanisms.ops import expit, exprelr


class l(State):

    USEQ10()

    PARAMETER({
        'zetal': 2.0,
        'gml': 1.0,
        'vhalfl': -61.0,
        'a0l': 0.001
    })
