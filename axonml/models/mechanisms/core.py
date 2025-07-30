from typing import List, Tuple

import torch


def coupled(state):
    return getattr(state, "coupled", False)


class State:
    is_q10 = False
    coupled = False
    _derivative: Tuple[str, bool] = None
    _assigned = set()
    _initialized = set()
    _buffers = set()
    _params = {}
    _diffusion: Tuple[float, str] = None


@torch.jit.interface
class MechanismInterface:
    def get(self, s: str) -> torch.Tensor:
        pass


def validate(mechanism):
    for v in mechanism._write_ion.values():
        for i in v:
            if not callable(getattr(mechanism, i, None)):
                raise ValueError(f"current {v} not implemented")
    return True


class Mechanism:
    _states = set()
    _ions = set()
    _range = set()
    _assigned = set()

    _params = {}
    _conductances = {}
    _init = {}
    _currents = {}

    _read_ion = {}
    _write_ion = {}
    _write_ion_c = {}
