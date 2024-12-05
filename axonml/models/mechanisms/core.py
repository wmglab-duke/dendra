from typing import Dict

import torch

from ..mixins import Parameterized


class State(Parameterized):
    
    is_q10 = False
    
    def __init__(self, temp: float):
        super().__init__()
        self.temp : float = temp
        if self.is_q10:
            assert getattr(self, 'calc_q10', None) is not None, "must implement `calc_q10` if using q10"
            self.register_buffer("q10_cache", self.calc_q10())

    def eval(self):
        if self.is_q10:
            self.q10_cache = self.calc_q10()
        return super().eval()

    def q10(self):
        if not self.training:
            return self.q10_cache
        return self.calc_q10()
    
    def inf(self, v):
        return self.alpha(v) / (self.alpha(v) + self.beta(v))

    def cnexp(self, gv, inf, tau_inv, dt):
        return inf - (inf - gv) * torch.exp(-dt * tau_inv)

    
class Mechanism(Parameterized):

    _states = set()
    _conductances = {}
    _init = {}
    
    def __init__(self, temp: float, v_init: float):
        super().__init__()
        self.temp : float = temp
        self.v_init : float = v_init
        self.states: Dict[str, torch.Tensor] = {}
        self.conductances: Dict[str, torch.Tensor] = {}
        self.derivatives = torch.nn.ModuleDict({
            cls.__name__: cls(self.temp) for cls in self._states
        })
        self._init_c()
        self._init_buffers(v_init)

    def _init_c(self):
        for n, v in self._conductances.items():
            self.conductances[n] = torch.tensor(v, device=self.device())

    def _init_buffers(self, v_init):
        for n, m in self.derivatives.items():
            if n in self._init:
                buffer_tensor = torch.tensor(self._init[n], device=self.device())
            else:
                buffer_tensor = m.inf(v_init)
            self.states[n] = buffer_tensor

    def inflate(self, v, area):
        self._inflate_c(v, area)
        self._inflate_s(v)

    def _inflate_c(self, v, area):
        for k, s in self.conductances.items():
            self.conductances[k] = (s*area)[:, None, None]

    def _inflate_s(self, v):
        for k, s in self.states.items():
            self.states[k] = s.expand(v.shape)

    def _advance(self, v, dt):
        for name, m in self.derivatives.items():
            self.states[name] = m.advance(self.states[name], v, dt)

    @torch.jit.export
    def get(self, s: str) -> torch.Tensor:
        return self.states[s]

    def i(self, v):
        raise NotImplementedError()
    