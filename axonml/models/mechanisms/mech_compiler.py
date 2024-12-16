from typing import Dict, List
import inspect
import linecache
import textwrap
import ast

import torch

from .compile_f import convert_func
from .core import Mechanism
from ..mixins import to_param

from .state_compiler import compile_state


def indent(text, level=0):
    return textwrap.indent(text, " " * (4 * level))


template = """
class mech(torch.nn.Module):
    _init_params: Dict[str, float]
    def __init__(self, temp, n_ax, n_nodes, name: str, params, read_ion, states, conductances, init, ic: dict = None, **kwargs):
        super().__init__()
        self.instantiate_parameters(params, **kwargs)
        self.temp = temp
        self._name = name
        self.DE = torch.nn.ModuleDict(
            {{state._name: state for state in states}}
        )

        for n, _ in self.DE.items():
            self.register_buffer(n, torch.tensor(0.0))
                
        self._init_params: Dict[str, float] = {{k: v for k, v in init.items()}}

        if ic is not None:
            self._init_params.update(ic)

        self.n_ax = n_ax
        self.n_nodes = n_nodes
        self.read_ion = read_ion

{state_buffer_assignments}

{current_buffer_assignments}

{assigned}

    def register_ion(self, ion):
        name = ion.name
        if name in self.read_ion:
            for v in self.read_ion[name]:
                self.register_buffer(v, getattr(ion, v))

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

    def _init_buffers_s(self, v_init):
        self.initial(v_init)
{init_state_buffers}
        return

    @torch.jit.ignore
    def set(self, key: str, value):
        p = getattr(self, key)
        if isinstance(p, torch.Tensor):
            p.data = torch.as_tensor(value, dtype=p.data.dtype, device=p.device)

    @torch.jit.ignore
    def get(self, key: str):
        return getattr(self, key)

    def _advance(self, v, dt):
{advance}
        return

{initial_f}

{breakpoint_f}

{current_equations}

{gtot}
"""

init_state_buffer_template = """
if '{state}' in self._init_params:
    buffer_tensor = torch.tensor(self._init_params['{state}'], device=v_init.device, dtype=v_init.dtype)
else:
    buffer_tensor = self.DE['{state}'].inf(v_init)
self.{state}[:] = buffer_tensor
"""

def extract_multipliers(class_def_str: str) -> List[str]:
    """
    Extracts all expressions that precede any expression matching '* (v - <x>)'
    within the given Python class definition string. The <x> can be a variable or an attribute.

    Args:
        class_def_str (str): The string representation of the Python class.

    Returns:
        List[str]: A list of multiplier expressions as strings.
    """

    class MultiplierVisitor(ast.NodeVisitor):
        def __init__(self):
            self.multipliers = []

        def is_target_subtraction(self, node: ast.BinOp) -> bool:
            """
            Checks if the given BinOp node represents a subtraction of the form (v - <x>),
            where <x> can be a Name or an Attribute.

            Args:
                node (ast.BinOp): The binary operation node to check.

            Returns:
                bool: True if the node matches the pattern (v - <x>), False otherwise.
            """
            if not isinstance(node, ast.BinOp):
                return False
            if not isinstance(node.op, ast.Sub):
                return False

            # Check if left operand is 'v'
            if not (isinstance(node.left, ast.Name) and node.left.id == 'v'):
                return False

            # Check if right operand is a Name or Attribute
            if isinstance(node.right, (ast.Name, ast.Attribute)):
                return True

            return False

        def visit_BinOp(self, node):
            # Check if the operation is multiplication
            if isinstance(node.op, ast.Mult):
                # Check the right operand for (v - x)
                if isinstance(node.right, ast.BinOp) and self.is_target_subtraction(node.right):
                    multiplier_expr = node.left
                    multiplier_str = ast.unparse(multiplier_expr).strip()
                    self.multipliers.append(multiplier_str)

                # Check the left operand for (v - x)
                elif isinstance(node.left, ast.BinOp) and self.is_target_subtraction(node.left):
                    multiplier_expr = node.right
                    multiplier_str = ast.unparse(multiplier_expr).strip()
                    self.multipliers.append(multiplier_str)

            # Continue traversing the AST
            self.generic_visit(node)

    # Parse the class definition string into an AST
    try:
        tree = ast.parse(class_def_str)
    except SyntaxError as e:
        print(f"SyntaxError while parsing the class definition: {e}")
        return []

    # Initialize and run the visitor
    visitor = MultiplierVisitor()
    visitor.visit(tree)

    return visitor.multipliers


def gtot(mechanism):
    functions = inspect.getmembers(mechanism, predicate=inspect.isfunction)
    ret = []
    for fname, f in functions:
        if fname in ["gtot"]:
            ret.append(inspect.getsource(f))
            return "\n".join(ret)
    multipliers = extract_multipliers(inspect.getsource(mechanism))
    if multipliers:
        sum_gtot = " + ".join(multipliers)
        return f"    def gtot(self): return {sum_gtot}"
    return "    def gtot(self): return torch.tensor(0.0)"


def conductances_init_assignments(conductances):
    assignments = []
    for k, _ in conductances.items():
        assignments.append(f"self.{k}_init = to_param(conductances['{k}'])")
    return "\n".join(assignments)


def conductances_buffer_assignments(conductances):
    assignments = []
    for k, _ in conductances.items():
        assignments.append(f"self.register_buffer('{k}', torch.tensor(0.0))")
    return "\n".join(assignments)


def state_buffer_assignments(states):
    assignments = []
    for k in states:
        name = k.__name__
        assignments.append(
            f"self.register_buffer('{name}', torch.zeros((n_ax, 1, n_nodes)))"
        )
    return "\n".join(assignments)


def current_buffer_assignments(currents, range_vars):
    assignments = []
    for k in currents:
        if k in range_vars:
            assignments.append(
                f"self.register_buffer('{k}_', torch.zeros((n_ax, 1, n_nodes)))"
            )  # noqa(0.0))")
    return "\n".join(assignments)


def init_state_buffers(states):
    assignments = []
    for k in states:
        name = k.__name__
        assignments.append(init_state_buffer_template.format(state=name))
    return "\n".join(assignments)


def inflate_states(states):
    assignments = []
    for k in states:
        name = k.__name__
        assignments.append(f"self.{name} = self.{name}.expand(v.shape)")
    return "\n".join(assignments)


def init_conductance_buffers(conductances):
    assignments = []
    for k, _ in conductances.items():
        assignments.append(f"self.{k} = self.{k}_init * area")
    return "\n".join(assignments)


def read_ion_buffers(read_ion):
    assignments = []
    for k, v in read_ion.items():
        for v_ in v:
            assignments.append(f"self.register_buffer('{v_}', torch.tensor(0.0))")
    return "\n".join(assignments)


def advance(states):
    assignments = []
    for k in states:
        name = k.__name__
        assignments.append(
            f"self.{name} = self.DE['{name}'].advance(self.{name}, v, dt)"
        )
    return "\n".join(assignments)


def current_equations(currents, mechanism, range_vars):
    assignments = []
    for k in currents:
        assign = k in range_vars
        assignments.append(convert_func(getattr(mechanism, k), assign))
    return "\n".join(assignments)


def load(m, attr):
    try:
        return getattr(m, attr)
    except AttributeError:
        return getattr(Mechanism, attr)


def get_function_body_as_str(func):
    source_lines = inspect.getsourcelines(func)[0]  # Get source code as lines
    body_lines = source_lines[1:]  # Skip the first line (def line)
    body = "".join(body_lines)  # Combine into a single string
    return body


default_f = """
    def {fname}(self, v):
{ret}
"""


def translate_f(mechanism, fname):
    f = getattr(mechanism, fname, None)
    if f:
        body = get_function_body_as_str(f)
    else:
        body = indent("return", 2)
    return default_f.format(fname=fname, ret=body)


def assigned_str_f(assigned):
    assignments = []
    for a in assigned:
        assignments.append(f"self.register_buffer('{a}', torch.tensor(0.0))")
    return "\n".join(assignments)


def compile_mechanism(mechanism, temp, n_ax, n_nodes, ic=None, **kwargs):
    states = mechanism._states

    params = load(mechanism, "_params")
    conductances = load(mechanism, "_conductances")
    init = load(mechanism, "_init")
    currents = load(mechanism, "_currents")
    range_vars = load(mechanism, "_range")
    ions = load(mechanism, "_ions")
    assigned = load(mechanism, "_assigned")

    read_ion = load(mechanism, "_read_ion")
    write_ion = load(mechanism, "_write_ion")
    write_ion_c = load(mechanism, "_write_ion_c")

    current_eqs = []
    for k, v in currents.items():
        current_eqs.extend(v)
    for _, v in write_ion.items():
        current_eqs.extend(v)

    conductances_init_assignments_str = conductances_init_assignments(
        mechanism._conductances
    )
    conductances_init_assignments_str = indent(conductances_init_assignments_str, 2)

    conductances_buffer_assignments_str = conductances_buffer_assignments(
        mechanism._conductances
    )
    conductances_buffer_assignments_str = indent(conductances_buffer_assignments_str, 2)

    state_buffer_assignments_str = state_buffer_assignments(states)
    state_buffer_assignments_str = indent(state_buffer_assignments_str, 2)

    current_buffer_assignments_str = current_buffer_assignments(current_eqs, range_vars)
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

    current_equations_str = current_equations(current_eqs, mechanism, range_vars)
    current_equations_str = indent(current_equations_str, 1)

    assigned_str = assigned_str_f(assigned)
    assigned_str = indent(assigned_str, 2)

    forward_str = template.format(
        state_buffer_assignments=state_buffer_assignments_str,
        current_buffer_assignments=current_buffer_assignments_str,
        read_ion_buffers=read_ion_buffers_str,
        assigned=assigned_str,
        init_state_buffers=init_state_buffers_str,
        advance=advance_str,
        current_equations=current_equations_str,
        breakpoint_f=translate_f(mechanism, "breakpoint"),
        initial_f=translate_f(mechanism, "initial"),
        gtot=gtot(mechanism),
    )

    filename = f"<{mechanism.__name__}_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)
    name = mechanism.__name__

    states = [compile_state(s, temp) for s in states]

    m = locals()["mech"](
        temp,
        n_ax,
        n_nodes,
        name,
        params,
        read_ion,
        states,
        conductances,
        init,
        ic=ic,
        **kwargs,
    )

    return m
