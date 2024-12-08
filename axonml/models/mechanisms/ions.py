from typing import Dict
import sys

import torch

from .declarations import add_to_namespace, add_to_namespace_dict


def USEION(ion, read=[], write=[]):
    frame = sys._getframe(1)
    namespace = frame.f_locals

    dct = {
        r: ion for r in read
    }

    add_to_namespace(namespace, '_ions', *[ion])
    add_to_namespace(namespace, '_states', *write)
    add_to_namespace_dict(namespace, '_read_ion', dct)


class Ion(torch.jit.ScriptModule):
    buffers: Dict[str, torch.Tensor]
    def __init__(self):
        super().__init__()
        self.buffers = {}

        self.ename = 'e'+self.__class__.__name__
        self.cname_o = self.__class__.__name__+'o'
        self.cname_i = self.__class__.__name__+'i'
        self.iname = 'i'+self.__class__.__name__

        self.c_style = 0
        self.e_style = 0
        self.einit = 0
        self.eadvance = 0
        self.cinit = 0

    def set_style(self, c_style, e_style, einit, eadvance, cinit):
        self.c_style = c_style
        self.e_style = e_style
        self.einit = einit
        self.eadvance = eadvance
        self.cinit = cinit

    @torch.jit.script_method
    def initialize(self, v):
        self.buffers[self.ename] = torch.as_tensor(self.buffers[self.ename], device=v.device, dtype=v.dtype)
        pass

    @torch.jit.script_method
    def advance(self, v, dt):
        pass

    @torch.jit.script_method
    def get(self, s: str) -> torch.Tensor:
        return self.buffers[s]
    
    @torch.jit.script_method
    def set(self, s: str, v: torch.Tensor) -> None:
        self.buffers[s] = v


class na(Ion):
    z = 1.0
    def __init__(self):
        super().__init__()
        self.buffers = {
            "ena": torch.tensor(1.0)
        }


class k(Ion):
    z = 1.0
    def __init__(self):
        super().__init__()
        self.buffers = {
            "ek": torch.tensor(1.0)
        }


class ca(Ion):
    z = 2.0
    def __init__(self):
        super().__init__()
        self.buffers = {
            "eca": torch.tensor(1.0)
        }
    

IONS = {
    'na': na,
    'k': k,
    'ca': ca
}