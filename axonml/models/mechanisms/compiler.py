import linecache
import textwrap

def indent(text, level=0):
    return textwrap.indent(text, ' '*(4*level))

import torch

from .compile_f import convert_func
from ..mixins import to_param


template = """
class mech(torch.nn.Module):
    def __init__(self, temp, params, states, conductances, init, ic: dict = None, **kwargs):
        super().__init__()
        self.instantiate_parameters(params, **kwargs)
        self.temp = temp
        self.DE = torch.nn.ModuleDict(
            {{cls.__name__: cls(self.temp) for cls in states}}
        )

        for n, _ in self.DE.items():
            self.register_buffer(n, torch.tensor(0.0))
                
        self._init_params: Dict[str, float] = {{k: v for k, v in init.items()}}

        if ic is not None:
            self._init_params.update(ic)

{conductances_init_assignments}

{conductances_buffer_assignments}

{state_buffer_assignments}

{current_buffer_assignments}

{read_ion_buffers}

    def instantiate_parameters(self, params, **kwargs):
        if params is not None:
            params = dict((k, kwargs.get(k, v)) for k, v in params.items())
            for name, value in params.items():
                if isinstance(value, dict):
                    setattr(self, name, [])
                    for pname, pval in value.items():
                        setattr(self, pname, to_param(pval))
                        getattr(self, name).append(getattr(self, pname))
                else:
                    setattr(self, name, to_param(value))

    def _init_buffers_s(self, v_init, area):
{init_state_buffers}
{init_conductance_buffers}
    
    @torch.jit.ignore
    def set(self, key: str, value):
        p = getattr(self, key)
        if isinstance(p, torch.Tensor):
            p.data = torch.as_tensor(value, dtype=p.data.dtype, device=p.device)

    @torch.jit.ignore
    def get(self, key: str):
        return getattr(self, key)

    def _inflate_s(self, v):
{inflate_states}

    def _advance(self, v, dt):
{advance}

{current_equations}
"""

init_state_buffer_template = """
if '{state}' in self._init_params:
    buffer_tensor = torch.tensor(self._init_params['{state}'], device=v_init.device, dtype=v_init.dtype)
else:
    buffer_tensor = self.DE['{state}'].inf(v_init)
self.{state} = buffer_tensor
"""




def conductances_init_assignments(conductances):
    assignments = []
    for k, _ in conductances.items():
        assignments.append(f"self.{k}_init = to_param(conductances['{k}'])")
    return '\n'.join(assignments)


def conductances_buffer_assignments(conductances):
    assignments = []
    for k, _ in conductances.items():
        assignments.append(f"self.register_buffer('{k}', torch.tensor(0.0))")
    return '\n'.join(assignments)


def state_buffer_assignments(states):
    assignments = []
    for k in states:
        name = k.__name__
        assignments.append(f"self.register_buffer('{name}', torch.tensor(0.0))")
    return '\n'.join(assignments)


def current_buffer_assignments(currents):
    assignments = []
    for k in currents:
        assignments.append(f"self.register_buffer('{k}_', torch.tensor(0.0))")
    return '\n'.join(assignments)


def init_state_buffers(states):
    assignments = []
    for k in states:
        name = k.__name__
        assignments.append(init_state_buffer_template.format(state=name))
    return '\n'.join(assignments)


def inflate_states(states):
    assignments = []
    for k in states:
        name = k.__name__
        assignments.append(f"self.{name} = self.{name}.expand(v.shape)")
    return '\n'.join(assignments)


def init_conductance_buffers(conductances):
    assignments = []
    for k, _ in conductances.items():
        assignments.append(f"self.{k} = self.{k}_init * area[:, None, None]")
    return '\n'.join(assignments)


def read_ion_buffers(read_ion):
    assignments = []
    for k, v in read_ion.items():
        for v_ in v:
            assignments.append(f"self.register_buffer('{v_}', torch.tensor(0.0))")
    return '\n'.join(assignments)


def advance(states):
    assignments = []
    for k in states:
        name = k.__name__
        assignments.append(f"self.{name} = self.DE['{name}'].advance(self.{name}, v, dt)")
    return '\n'.join(assignments)


def current_equations(currents, mechanism, range_):
    assignments = []
    for k in currents:
        assign = (k in range_)
        assignments.append(convert_func(getattr(mechanism, k), assign))
    return '\n'.join(assignments)



def compile_mechanism(mechanism, temp, ic=None, **kwargs):

    states = mechanism._states
    params = mechanism._params
    conductances = mechanism._conductances
    init = mechanism._init
    currents = mechanism._currents
    _range = mechanism._range
    ions = mechanism._ions

    read_ion = mechanism._read_ion
    write_ion = mechanism._write_ion
    write_ion_c = mechanism._write_ion_c

    current_eqs = []
    for k, v in currents.items():
        current_eqs.extend(v)
    for _, v in write_ion.items():
        current_eqs.extend(v)

    conductances_init_assignments_str = conductances_init_assignments(mechanism._conductances)
    conductances_init_assignments_str = indent(conductances_init_assignments_str, 2)

    conductances_buffer_assignments_str = conductances_buffer_assignments(mechanism._conductances)
    conductances_buffer_assignments_str = indent(conductances_buffer_assignments_str, 2)

    state_buffer_assignments_str = state_buffer_assignments(states)
    state_buffer_assignments_str = indent(state_buffer_assignments_str, 2)

    current_buffer_assignments_str = current_buffer_assignments(current_eqs)
    current_buffer_assignments_str = indent(current_buffer_assignments_str, 2)

    read_ion_buffers_str = read_ion_buffers(read_ion)
    read_ion_buffers_str = indent(read_ion_buffers_str, 2)    

    inflate_states_str = inflate_states(states)
    inflate_states_str = indent(inflate_states_str, 2)

    init_state_buffers_str = init_state_buffers(states)
    init_state_buffers_str = indent(init_state_buffers_str, 2)

    init_conductance_buffers_str = init_conductance_buffers(conductances)
    init_conductance_buffers_str = indent(init_conductance_buffers_str, 2)

    advance_str = advance(states)
    advance_str = indent(advance_str, 2)

    current_equations_str = current_equations(current_eqs, mechanism, _range)
    current_equations_str = indent(current_equations_str, 1)

    forward_str = template.format(
        conductances_init_assignments=conductances_init_assignments_str,
        conductances_buffer_assignments=conductances_buffer_assignments_str,
        state_buffer_assignments=state_buffer_assignments_str,
        current_buffer_assignments=current_buffer_assignments_str,
        read_ion_buffers=read_ion_buffers_str,
        init_state_buffers=init_state_buffers_str,
        init_conductance_buffers=init_conductance_buffers_str,
        inflate_states=inflate_states_str,
        advance=advance_str,
        current_equations=current_equations_str
    )

    print(forward_str)

    filename = "<compiler_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    m = torch.jit.script(locals()["mech"](temp, params, states, conductances, init, ic=ic, **kwargs))

    return m