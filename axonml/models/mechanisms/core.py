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


@torch.jit.interface
class MechanismInterface:
    def get(self, s: str) -> torch.Tensor:
        pass

    
class Mechanism(Parameterized):

    _states = set()
    _conductances = {}
    _init = {}

    _init_params: Dict[str, float]
    states: Dict[str, torch.Tensor]
    
    def __init__(self, temp: float, v_init: float, **kwargs):
        super().__init__()
        self.temp : float = temp
        self.v_init : float = v_init

        self.states: Dict[str, torch.Tensor] = {}
        self.dynamics = torch.nn.ModuleDict({
            cls.__name__: cls(self.temp) for cls in self._states
        })

        # -- bunch of stuff to handle initial conditions + torch compiler --
        self._init_params : Dict[str, float] = {
            k:v for k, v in self._init.items()
        }
        self.init(v_init)
        for k, v in kwargs.items():
            self.set(k, v)

    def set(self, key, value):
        p = getattr(self, key)
        if isinstance(p, torch.Tensor):
            p.data = torch.tensor(value, dtype=p.data.dtype, device=p.device)

    def _init_buffers_s(self, v_init):
        for n, m in self.dynamics.items():
            if n in self._init_params:
                buffer_tensor = torch.tensor(self._init_params[n], device=v_init.device)
            else:
                buffer_tensor = m.inf(v_init)
            self.states[n] = buffer_tensor

    @torch.jit.export
    def init(self, v_init):
        self._init_buffers_s(v_init)

    @torch.jit.export
    def inflate(self, v):
        self._inflate_s(v)

    @torch.jit.export
    def _inflate_s(self, v):
        for k, s in self.states.items():
            self.states[k] = s.expand(v.shape)

    def _advance(self, v, dt):
        for name, m in self.dynamics.items():
            self.states[name] = m.advance(self.states[name], v, dt)

    @torch.jit.export
    def get(self, s: str) -> torch.Tensor:
        return self.states[s]

    def i(self, v):
        raise NotImplementedError()
    