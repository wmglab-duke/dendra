import inspect
from typing import Dict, List, Tuple
import re
import types
import linecache

from nmodl.ode import integrate2c
import torch

from ..mixins import Parameterized

forward_template = """
def forward(self, t: Dict[str, torch.Tensor]) -> torch.Tensor:
  {populate_input_globals}
  {f}
  return {state}
"""

input_globals_template="""
  {field} = t["{field}"]
"""

def make_integrate(state):
    input_fields = state.all_names()

    input_globals = "".join(input_globals_template.format(field=field) for field in input_fields)

    state_name = state.__class__.__name__

    forward_str = forward_template.format(
        f=state.i_func,
        populate_input_globals=input_globals,
        state=state_name,
    )

    class ModuleTemplate(torch.nn.Module):
        def forward(self):
            raise NotImplementedError()

    filename = "<forward_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + '\n' for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    m = ModuleTemplate()
    m.forward = types.MethodType(locals()["forward"], m)
    m = torch.jit.script(m)
    return m

# Get a list of all available PyTorch operations
torch_operations = set(dir(torch))

# Function to modify the input string
def modify_operations(input_string):
    # Regular expression to find function names and calls
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    
    # Function to replace matches with 'torch.' prefix if they are PyTorch operations
    def replacer(match):
        func_name = match.group(1)
        if func_name in torch_operations:
            return f"torch.{func_name}("
        return match.group(0)
    
    # Apply the replacement
    modified_string = re.sub(pattern, replacer, input_string)
    return modified_string


class State(Parameterized):
    is_q10 = False
    _derivative : str = None

    def __init__(self, temp: float):
        super().__init__()
        assert self._derivative is not None, "must implement DERIVATIVE"
        self.temp: float = temp
        if self.is_q10:
            assert (
                getattr(self, "calc_q10", None) is not None
            ), "must implement `calc_q10` if using q10"
            self.register_buffer("q10_cache", self.calc_q10())
        
        self._name = self.__class__.__name__
        self._export_names = self.export_names()
        self._all_names = self.all_names()

        self.i_func = modify_operations(integrate2c(self._derivative, "dt", self._export_names))
        self.integrate = make_integrate(self)

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

    def set(self, key, value):
        p = getattr(self, key)
        if isinstance(p, torch.Tensor):
            p.data = torch.tensor(value, dtype=p.data.dtype, device=p.device)

    def __setattr__(self, name, param):
        if name in self._parameters:
            p = self._parameters[name]
            param = torch.nn.Parameter(torch.tensor(param, device=p.device, dtype=p.dtype))
        return super().__setattr__(name, param)
    
    def export(self, v) -> Tuple[torch.Tensor]:
        pass

    def export_names(self):
        return_lines = list()
        for line in inspect.getsourcelines(self.export)[0]:
            line = line.strip()
            if line.startswith("return"):
                return_lines = list(line[7:].split(", "))
                break
        return return_lines
    
    def all_names(self) -> List[str]:
        return [self._name, "dt"] + self._export_names
    
    def advance(self, state, v, dt):
        export = self.export(v)
        values = [state, dt, *export]
        d = {n:v for n, v in zip(self._all_names, values)}
        return self.integrate(d)

@torch.jit.script
def make_dict(names: List[str], values: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    d = {n:v for n, v in zip(names, values)}
    return d

@torch.jit.interface
class MechanismInterface:
    def get(self, s: str) -> torch.Tensor:
        pass


class Mechanism(Parameterized):
    _states = set()
    _ions = set()
    _conductances = {}
    _init = {}
    _read_ion = {}

    _init_params: Dict[str, float]
    states: Dict[str, torch.Tensor]

    def __init__(self, temp: float, v_init: float, ic: dict = None, **kwargs):
        super().__init__()
        self.temp: float = temp
        self.v_init: float = v_init

        self.states: Dict[str, torch.Tensor] = {}
        self.DE = torch.nn.ModuleDict(
            {cls.__name__: cls(self.temp) for cls in self._states}
        )
        self.ions = torch.nn.ModuleDict()

        # -- bunch of stuff to handle initial conditions + torch compiler --
        self._init_params: Dict[str, float] = {k: v for k, v in self._init.items()}
        if ic is not None:
            self._init_params.update(ic)
        self._init_buffers_s(v_init)
        self.init(v_init)
        for k, v in kwargs.items():
            self.set(k, v)

    def register_ion(self, name, ion):
        self.ions[name] = ion

    def set(self, key, value):
        p = getattr(self, key)
        if isinstance(p, torch.Tensor):
            p.data = torch.tensor(value, dtype=p.data.dtype, device=p.device)

    def __setattr__(self, name, param):
        if name in self._parameters:
            p = self._parameters[name]
            param = torch.nn.Parameter(torch.tensor(param, device=p.device, dtype=p.dtype))
        return super().__setattr__(name, param)
    
    def __getattr__(self, name):
        if name in self._read_ion:
            return self.ions[self._read_ion[name]].get(name)
        return super().__getattr__(name)

    def _init_buffers_s(self, v_init):
        for n, m in self.DE.items():
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
        for name, m in self.DE.items():
            self.states[name] = m.advance(self.states[name], v, dt)

    @torch.jit.export
    def get(self, s: str) -> torch.Tensor:
        return self.states[s]

    def i_na(self, v):
        return None
    
    def i_k(self, v):
        return None
    
    def i_ca(self, v):
        return None

    def i(self, v):
        return None
