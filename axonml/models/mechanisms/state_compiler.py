from typing import Dict
import inspect
import linecache
import textwrap
import string
import random

import torch

from .core import State
from ..mixins import to_param

from nmodl.ode import integrate2c

import re
from .ops import *


def extract_vars(f, exclude):
    pattern = r'\b[a-zA-Z_]\w*\b'
    all_variables = re.findall(pattern, f)
    filtered_variables = [var for var in all_variables if var not in exclude]
    return filtered_variables


def indent(text, level=0):
    return textwrap.indent(text, " " * (4 * level))


def replace(input_string, replace_list):
    for substring in replace_list:
        input_string = re.sub(rf'\b{substring}\b', f'self.{substring}', input_string)
    return input_string


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


def convert(deriv, state, assigned, use_pade_approx=False):
    v = extract_vars(deriv, set([state]) | assigned)
    f = integrate2c(deriv, "dt", v, use_pade_approx=use_pade_approx)
    return modify_operations(replace(f, v))


template = """
class state(torch.nn.Module):
    def __init__(self, temp, is_q10: bool, name: str, params):
        super().__init__()
        self.instantiate_parameters(params)
        self.temp = temp
        self._name = name
        self.is_q10 = is_q10
        if self.is_q10:
            self.register_buffer("q10_cache", self.calc_q10())

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

    @torch.jit.ignore
    def set(self, key: str, value):
        p = getattr(self, key)
        if isinstance(p, torch.Tensor):
            p.data = torch.as_tensor(value, dtype=p.data.dtype, device=p.device)

    @torch.jit.ignore
    def get(self, key: str):
        return getattr(self, key)

    def advance(self, {state}, v, dt):
        return self.integrate({state}, dt {breakpoint_int})

    def integrate(self, {state}, dt {integrate_args}):
        {integrate_f}
        return {state}

{inf_f}

{breakpoint_f}

{calc_q10_f}

{helpers}
"""
    

def assigned_str_f(assigned):
    assignments = []
    for a in assigned:
        assignments.append(f"self.register_buffer('{a}', torch.tensor(0.0))")
    return "\n".join(assignments)


def load(m, attr):
    try:
        return getattr(m, attr)
    except AttributeError:
        return getattr(State, attr)
    

def randomword(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def get_function_body_as_str(func):
    source_lines = inspect.getsourcelines(func)[0]  # Get source code as lines
    body_lines = source_lines[1:]  # Skip the first line (def line)
    body = "".join(body_lines)  # Combine into a single string
    return body


default_f = """
    def {fname}(self, v):
{ret}
"""

breakpoint_str = """
    def breakpoint(self, v):
{body}
{ret}
"""


def translate_breakpoint(state, assigned):
    f = getattr(state, "breakpoint", None)
    if not f:
        return ""
    body = get_function_body_as_str(f)
    return breakpoint_str.format(body=body, ret=indent("return " + ", ".join(assigned), 2))


def translate_f(mechanism, fname, default=None):
    f = getattr(mechanism, fname, None)
    if f:
        body = get_function_body_as_str(f)
    else:
        if default is None:
            default = "return"
        body = indent(default, 2)
    return default_f.format(fname=fname, ret=body)


def calc_q10(s, is_q10):
    if not is_q10:
        return ""
    q10_f = getattr(s, "calc_q10", None)
    if q10_f is None:
        raise ValueError("Must specify calc_q10 function")
    return inspect.getsource(q10_f)


def collect_helper_functions(state):
    functions = inspect.getmembers(state, predicate=inspect.isfunction)
    ret = []
    for fname, f in functions:
        if fname not in ["breakpoint", "inf", "calc_q10"]:
            ret.append(inspect.getsource(f))
    return "\n".join(ret)


def breakpoint_int(s):
    f = getattr(s, "breakpoint", None)
    if not f:
        return ""
    return ", *self.breakpoint(v)"


def integrate_args(assigned):
    if not assigned:
        return ""
    return ", " + ", ".join(assigned)


def compile_state(s, temp):
    state_name = s.__name__
    params = load(s, "_params")
    assigned = load(s, "_assigned")
    assigned_list = list(assigned)
    is_q10 = load(s, "is_q10")
    derivative = load(s, "_derivative")

    if derivative is None:
        raise ValueError("Must specify derivative function")

    assigned_str = assigned_str_f(assigned)
    assigned_str = indent(assigned_str, 2)

    integrate_f = convert(derivative[0], state_name, assigned, use_pade_approx=derivative[1])

    forward_str = template.format(
        # assigned=assigned_str,
        state=state_name,
        breakpoint_int=breakpoint_int(s),
        integrate_f=integrate_f,
        breakpoint_f=translate_breakpoint(s, assigned_list),
        integrate_args=integrate_args(assigned_list),
        inf_f=translate_f(s, "inf", "return self.alpha(v) / (self.alpha(v) + self.beta(v))"),
        calc_q10_f=calc_q10(s, is_q10),
        helpers=collect_helper_functions(s),
    )

    print(forward_str)

    filename = f"<{state_name}_{randomword(5)}_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    m = torch.jit.script(
        locals()["state"](temp, is_q10, state_name, params)
    )

    return m
