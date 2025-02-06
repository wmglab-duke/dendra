from typing import Dict, List
import inspect
import linecache
import textwrap
import ast
import re
import math

import warnings
from sympy import symbols, sympify, Poly, expand, factor


import torch

from .compile_f import convert_func
from .core import Mechanism, coupled
from ..mixins import to_param
from .ops import *
from axonml import const

from .handler.defaults import valid_concentrations
from .state_compiler import compile_state, compile_coupled_state


def indent(text, level=0):
    return textwrap.indent(text, " " * (4 * level))


def replace_v(code_str):
    # Use a regex with word boundaries to ensure only standalone 'v' is replaced.
    # The replacement inserts '(v + v_n) / 2' in place of v.
    return re.sub(r"\bv\b", "(v + v_n) / 2", code_str)


def factor_linear_in_x_from_codeblock(code_str, x_var="v_n"):
    lines = code_str.strip().split("\n")

    # Identify self-prefixed variables
    pattern = r"self\.(\w+)"
    self_vars_all = re.findall(pattern, code_str)
    self_vars_all = set(self_vars_all)
    self_mapping = {var: f"self.{var}" for var in self_vars_all}

    env = {}

    def parse_expr(expr_str):
        # Extract potential variables
        potential_vars = set(re.findall(r"[a-zA-Z_]\w*", expr_str))
        for var in potential_vars:
            if var not in env:
                env[var] = symbols(var, real=True)
        return sympify(expr_str, locals=env)

    final_expr = None

    # Parse line by line
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_no_self = line.replace("self.", "")

        if line_no_self.startswith("return "):
            return_expr_str = line_no_self[len("return ") :].strip()
            final_expr = parse_expr(return_expr_str)
        elif "=" in line_no_self:
            lhs, rhs = line_no_self.split("=", 1)
            var_name = lhs.strip()
            rhs_expr_str = rhs.strip()
            rhs_expr = parse_expr(rhs_expr_str)
            env[var_name] = rhs_expr
        else:
            final_expr = parse_expr(line_no_self)

    if final_expr is None:
        raise ValueError("No final expression or return statement found.")

    if x_var not in env:
        env[x_var] = symbols(x_var, real=True)
    x = env[x_var]

    # Factor the final_expr as A + B*x
    expr_expanded = expand(final_expr)
    p = Poly(expr_expanded, x)

    if p.degree() != 1:
        raise ValueError("Expression is not linear in x.")

    A = p.eval(0)
    B = p.coeff_monomial(x)

    # Now factor each of A and B individually
    A_factor = factor(A)
    B_factor = factor(B)

    # Convert to strings
    A_str = str(A_factor)
    B_str = str(B_factor)

    # Restore self. prefixes
    for var in sorted(self_mapping.keys(), key=len, reverse=True):
        A_str = re.sub(rf"\b{var}\b", self_mapping[var], A_str)
        B_str = re.sub(rf"\b{var}\b", self_mapping[var], B_str)

    return A_str, B_str


template = """
class {mech}(torch.nn.Module):
    _init_params: Dict[str, float]
    def __init__(self, temp, diameters, n_ax, n_nodes, name: str, params, distributions, read_ion, write_ion_c, states, conductances, init, ic: dict = None, mask=None):
        super().__init__()
        self.instantiate_parameters(params)
        self.instantiate_distributions(distributions)
        self.temp = temp
        self.register_buffer("diam", diameters.view(-1, 1, 1))
        self._name = name
        self.DE = torch.nn.ModuleDict(
            {{state._name: state for state in states}}
        )

        if mask is not None:
            mask_ = torch.ones(1, 1, n_nodes)
            mask_[:, :, mask] = 0
            self.register_buffer("mask", mask_)
                
        self._init_params: Dict[str, float] = {{k: v for k, v in init.items()}}

        if ic is not None:
            self._init_params.update(ic)

        self.n_ax = n_ax
        self.n_nodes = n_nodes
        self.read_ion = read_ion
        self.write_ion_c = write_ion_c

{state_buffer_assignments}

{distribution_buffer_assignments}

{current_buffer_assignments}

{assigned}

    def register_ion(self, ion):
        name = ion.name
        if name in self.read_ion:
            for v in self.read_ion[name]:
                self.register_buffer(v, getattr(ion, v))
                for _, s in self.DE.items():
                    s.register_buffer(v, getattr(ion, v))

        if name in self.write_ion_c:
            for v in self.write_ion_c[name]:
                self.register_buffer(v, getattr(ion, v))

    def instantiate_parameters(self, params):
        if params is not None:
            for name, value in params.items():
                if isinstance(value, dict):
                    setattr(self, name, [])
                    for pname, pval in value.items():
                        setattr(self, pname, to_param(pval))
                        getattr(self, name).append(getattr(self, pname))
                else:
                    setattr(self, name, to_param(value))

    def detach(self):
{detach}
        return

    def instantiate_distributions(self, distributions):
        if distributions is not None:
            for name, dist in distributions.items():
                setattr(self, name+"_d", dist)

    def _init_buffers_s(self, v_init):
{init_state_buffers}
{init_distribution_buffers}
        self.initial(v_init)
        for _, s in self.DE.items():
            s.initialize(v_init)
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

{coupled_infs}

{current_equations}

{gtot}
"""

init_state_buffer_template = """
if '{state}' in self._init_params:
    buffer_tensor = torch.tensor(self._init_params['{state}'], device=v_init.device, dtype=v_init.dtype)
else:
    buffer_tensor = self.DE['{state}'].inf(v_init)
self.{state}[:] = buffer_tensor
self.{state}.detach_()
"""

init_state_buffers_coupled_template = """
if '{state}' in self._init_params:
    buffer_tensor = torch.tensor(self._init_params['{state}'], device=v_init.device, dtype=v_init.dtype)
else:
    buffer_tensor = self.{state}_inf(v_init)
self.{state}[:] = buffer_tensor
self.{state}.detach_()
"""


mech_inf_template = """
def {state}_inf(self, v):
    return torch.tensor(0.0, device=v.device, dtype=v.dtype)
"""


distribution_init_template = """
self.{name} = self.{name}_d._sample(self.{name})
"""


def detach(states):
    assignments = []
    for k in states:
        if not k.coupled:
            name = k._name
            assignments.append(f"self.{name}.detach_()")
        else:
            for name in k._state_names:
                assignments.append(f"self.{name}.detach_()")
    return "\n".join(assignments)


def init_distribution_buffers(distributions):
    assignments = []
    for k, _ in distributions.items():
        assignments.append(distribution_init_template.format(name=k))
    return "\n".join(assignments)


def coupled_infs(mechanism, states):
    assignments = []
    for k in states:
        if k.coupled:
            for name in k._state_names:
                if name not in valid_concentrations():
                    f = getattr(mechanism, f"{name}_inf", None)
                    if f is not None:
                        assignments.append(inspect.getsource(f))
                    else:
                        assignments.append(
                            indent(mech_inf_template.format(state=name), 1)
                        )
    return "\n".join(assignments)


def init_state_buffers(states):
    assignments = []
    for k in states:
        if not k.coupled:
            name = k._name
            if name not in valid_concentrations():
                assignments.append(init_state_buffer_template.format(state=name))
        else:
            for name in k._state_names:
                if name not in valid_concentrations():
                    assignments.append(
                        init_state_buffers_coupled_template.format(state=name)
                    )
    return "\n".join(assignments)


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
            if not (isinstance(node.left, ast.Name) and node.left.id == "v"):
                return False

            # Check if right operand is a Name or Attribute
            if isinstance(node.right, (ast.Name, ast.Attribute)):
                return True

            return False

        def visit_BinOp(self, node):
            # Check if the operation is multiplication
            if isinstance(node.op, ast.Mult):
                # Check the right operand for (v - x)
                if isinstance(node.right, ast.BinOp) and self.is_target_subtraction(
                    node.right
                ):
                    multiplier_expr = node.left
                    multiplier_str = ast.unparse(multiplier_expr).strip()
                    self.multipliers.append(multiplier_str)

                # Check the left operand for (v - x)
                elif isinstance(node.left, ast.BinOp) and self.is_target_subtraction(
                    node.left
                ):
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


def state_buffer_assignments(states):
    assignments = []
    for k in states:
        if not k.coupled:
            name = k._name
            if name not in valid_concentrations():
                assignments.append(
                    f"self.register_buffer('{name}', torch.zeros((n_ax, 1, n_nodes)))"
                )
        else:
            names = k._state_names
            for name in names:
                if name not in valid_concentrations():
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


def init_conductance_buffers(conductances):
    assignments = []
    for k, _ in conductances.items():
        assignments.append(f"self.{k} = self.{k}_init * area")
    return "\n".join(assignments)


def advance(states):
    assignments = []
    for k in states:
        if k.coupled:
            assignments.append(coupled_assignment(k))
        else:
            name = k._name
            if name in valid_concentrations():
                assignments.append(
                    f"self.{name}[:] = self.DE['{name}'].advance(self.{name}, v, dt)"
                )
            else:
                assignments.append(
                    f"self.{name} = self.DE['{name}'].advance(self.{name}, v, dt)"
                )
    return "\n".join(assignments)


current_eq_template = """
def {k}(self, v):
    return {v}
"""

current_eq_template_assign = """
def {k}(self, v):
    self.{k}_ = {v}
    return self.{k}_
"""


current_tot_template = """
def {k}_tot(self, v):
{body}
"""


def multiply_return_value(code_string, multiplier_expr: str) -> str:
    """
    Given a function body in a string,
    replace any `return x` statement with `return <multiplier_expr> * x`.
    """
    # Pattern captures:
    # (1) the word 'return'
    # (2) optional whitespace
    # (3) the return expression (grouped as (.+) to capture it)
    pattern = r"(return)\s+(.+)"

    # Use an f-string to insert the multiplier expression
    # before whatever was captured in group 2.
    replacement = rf"return {multiplier_expr} * \2"

    # Perform the substitution.
    new_code = re.sub(pattern, replacement, code_string)
    return new_code


def current_equations(currents, mechanism, range_vars, df, mask):
    assignments = []
    unfactorable = [] if df else None
    for k in currents:
        assign = k in range_vars
        if not df:
            code = convert_func(getattr(mechanism, k), assign)
            if mask is not None:
                code = multiply_return_value(code, "self.mask")
            assignments.append(code)
            code_block = get_function_body_as_str(getattr(mechanism, k))
            if mask is not None:
                code_block = multiply_return_value(code_block, "self.mask")
            assignments.append(current_tot_template.format(k=k, body=code_block))
        else:
            code_block = get_function_body_as_str(getattr(mechanism, k))
            if mask is not None:
                code_block = multiply_return_value(code_block, "self.mask")
            assignments.append(current_tot_template.format(k=k, body=code_block))
            try:
                i, _ = factor_linear_in_x_from_codeblock(replace_v(code_block))
                if assign:
                    code = current_eq_template_assign.format(k=k, v=i)
                    if mask is not None:
                        code = multiply_return_value(code, "self.mask")
                    assignments.append(code)
                else:
                    code = current_eq_template.format(k=k, v=i)
                    if mask is not None:
                        code = multiply_return_value(code, "self.mask")
                    assignments.append(code)
            except:
                warnings.warn(
                    "Could not confirm all currents are linear in v. Dufort-Frankel may not be stable."
                )
                code = convert_func(getattr(mechanism, k), assign)
                if mask is not None:
                    code = multiply_return_value(code, "self.mask")
                assignments.append(code)
                unfactorable.append(k)
    return "\n".join(assignments), unfactorable


def gtot(currents, mechanism, df):
    if not df:
        return "    def gtot(self): return torch.tensor(0.0)", False
    assignments = []
    for k in currents:
        code_block = get_function_body_as_str(getattr(mechanism, k))
        try:
            _, b = factor_linear_in_x_from_codeblock(replace_v(code_block))
            assignments.append(b)
        except:
            pass
    has_gtot = True
    if not assignments:
        has_gtot = False
        return "    def gtot(self): return torch.tensor(0.0)", has_gtot
    s = " + ".join(assignments)
    return f"    def gtot(self): return {s}", has_gtot


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


def coupled_assignment(state):
    name = state._name
    states_names = state._state_names
    template = "{lhs} = self.DE['{name}'].advance({rhs}, v, dt)"
    lhs = []
    rhs = []
    for state_name in states_names:
        if not state_name in valid_concentrations():
            lhs.append(f"self.{state_name}")
            rhs.append(f"self.{state_name}")
        else:
            lhs.append(f"self.{state_name}[:]")
            rhs.append(f"self.{state_name}")
    lhs = ", ".join(lhs)
    rhs = ", ".join(rhs)
    return template.format(name=name, lhs=lhs, rhs=rhs)


def parse_params_distributions(params, kwargs):
    params = dict((k, kwargs.get(k, v)) for k, v in params.items())
    regular_params = {}
    distributions = {}
    for k, v in params.items():
        if isinstance(v, torch.nn.Module):
            distributions[k] = v
        else:
            regular_params[k] = v
    return regular_params, distributions


def distribution_buffers(distributions):
    assignments = []
    for k, v in distributions.items():
        assignments.append(
            f"self.register_buffer('{k}', torch.zeros((n_ax, 1, n_nodes)))"
        )
    return "\n".join(assignments)


def compile_mechanism(
    mechanism,
    temp,
    diameters,
    n_ax,
    n_nodes,
    df=False,
    ic=None,
    pade=None,
    mask=None,
    **kwargs,
):
    states = mechanism._states

    params = load(mechanism, "_params")
    params, distributions = parse_params_distributions(params, kwargs)
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

    states_compiled = []
    for s in states:
        if coupled(s):
            states_compiled.append(
                compile_coupled_state(s, temp, diameters, pade=pade, **kwargs)
            )
        else:
            states_compiled.append(
                compile_state(s, temp, diameters, pade=pade, **kwargs)
            )

    state_buffer_assignments_str = state_buffer_assignments(states_compiled)
    state_buffer_assignments_str = indent(state_buffer_assignments_str, 2)

    current_buffer_assignments_str = current_buffer_assignments(current_eqs, range_vars)
    current_buffer_assignments_str = indent(current_buffer_assignments_str, 2)

    init_state_buffers_str = init_state_buffers(states_compiled)
    init_state_buffers_str = indent(init_state_buffers_str, 2)

    init_conductance_buffers_str = init_conductance_buffers(conductances)
    init_conductance_buffers_str = indent(init_conductance_buffers_str, 2)

    advance_str = advance(states_compiled)
    advance_str = indent(advance_str, 2)

    current_equations_str, unfactorable = current_equations(
        current_eqs, mechanism, range_vars, df, mask
    )
    current_equations_str = indent(current_equations_str, 1)

    assigned_str = assigned_str_f(assigned)
    assigned_str = indent(assigned_str, 2)

    distribution_buffer_assignments_str = distribution_buffers(distributions)
    distribution_buffer_assignments_str = indent(distribution_buffer_assignments_str, 2)

    init_distribution_buffers_str = init_distribution_buffers(distributions)
    init_distribution_buffers_str = indent(init_distribution_buffers_str, 2)

    gtot_str, has_gtot = gtot(current_eqs, mechanism, df)

    forward_str = template.format(
        mech=mechanism.__name__,
        state_buffer_assignments=state_buffer_assignments_str,
        current_buffer_assignments=current_buffer_assignments_str,
        assigned=assigned_str,
        init_state_buffers=init_state_buffers_str,
        advance=advance_str,
        current_equations=current_equations_str,
        breakpoint_f=translate_f(mechanism, "breakpoint"),
        initial_f=translate_f(mechanism, "initial"),
        gtot=gtot_str,
        coupled_infs=coupled_infs(mechanism, states_compiled),
        distribution_buffer_assignments=distribution_buffer_assignments_str,
        init_distribution_buffers=init_distribution_buffers_str,
        detach=indent(detach(states_compiled), 2),
    )

    filename = f"<{mechanism.__name__}_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)
    name = mechanism.__name__

    m = locals()[mechanism.__name__](
        temp,
        diameters,
        n_ax,
        n_nodes,
        name,
        params,
        distributions,
        read_ion,
        write_ion_c,
        states_compiled,
        conductances,
        init,
        ic=ic,
        mask=mask,
    )

    return m, unfactorable, has_gtot
