import inspect
from typing import Dict, List, Tuple
import re
import linecache

from nmodl.ode import integrate2c
import torch

from ..mixins import Parameterized, to_param


forward_template = """
def advance({names}) -> torch.Tensor:
  {f}
  return {state}
"""


def make_integrate(state):
    input_fields = state._all_names
    state_name = state._name

    forward_str = forward_template.format(
        names=str(input_fields)[1:-1].replace("'", ""),
        f=state.i_func,
        state=state_name,
    )

    filename = "<forward_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    m = torch.jit.script(locals()["advance"])
    return m


# Get a list of all available PyTorch operations
torch_operations = set(dir(torch))


# Function to modify the input string
def modify_operations(input_string):
    # Regular expression to find function names and calls
    pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("

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
    _derivative: Tuple[str, bool] = None

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

        self.i_func = modify_operations(
            integrate2c(
                self._derivative[0],
                "dt",
                self._export_names,
                use_pade_approx=self._derivative[1],
            )
        )
        self.integrate = make_integrate(self)

    def eval(self):
        if self.is_q10:
            self.q10_cache = self.calc_q10()
            self.q10 = self.return_q10_cache
        return super().eval()

    def train(self):
        if self.is_q10:
            self.q10 = self.calc_q10
        return super().train()

    def return_q10_cache(self):
        return self.q10_cache

    def q10(self):
        if not self.training:
            return self.q10_cache
        return self.calc_q10()

    def inf(self, v):
        return self.alpha(v) / (self.alpha(v) + self.beta(v))

    def set(self, key, value):
        p = getattr(self, key)
        if isinstance(p, torch.Tensor):
            p.data = torch.tensor(value, dtype=p.data.dtype, device=p.device)

    def __setattr__(self, name, param):
        if name in self._parameters:
            p = self._parameters[name]
            param = torch.nn.Parameter(
                torch.tensor(param, device=p.device, dtype=p.dtype)
            )
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
        return self.integrate(state, dt, *export)


@torch.jit.interface
class MechanismInterface:
    def get(self, s: str) -> torch.Tensor:
        pass


def validate(mechanism):
    for v in mechanism._write_ion.values():
        if not callable(getattr(mechanism, v, None)):
            raise ValueError(f"current {v} not implemented")
    return True


class Mechanism:
    _states = set()
    _ions = set()
    _conductances = {}
    _init = {}
    _currents = {}
    _range = set()

    _read_ion = {}
    _write_ion = {}
    _write_ion_c = {}

    _init_params: Dict[str, float]
