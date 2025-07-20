import re
from typing import List, Tuple, Sequence
from types import MethodType
import torch
import ast

from .ode import integrate2c
from axonml.models.parametric import Parameterized
from axonml.helpers import DEBUG, PADE


# PyTorch operations
torch_operations = set(dir(torch))


class UnderscoreLHS(ast.NodeTransformer):
    """
    An AST NodeTransformer that traverses an AST and prepends an underscore
    to the variable names on the left-hand side of any assignment.
    """

    def _prefix_target(self, target_node):
        """Recursively prefixes the appropriate part of an assignment target."""
        if isinstance(target_node, ast.Name):
            # This is a simple variable name like 'a'.
            target_node.id = '_' + target_node.id
        elif isinstance(target_node, ast.Attribute):
            # This is an attribute like 'obj.value'. We change 'value' to '_value'.
            target_node.attr = '_' + target_node.attr
        elif isinstance(target_node, (ast.Tuple, ast.List)):
            # This is unpacking like 'a, b = ...'. Recurse on each element.
            for element in target_node.elts:
                self._prefix_target(element)
        elif isinstance(target_node, ast.Subscript):
            # This is an item assignment like 'd[k] = v'. Recurse on the variable 'd'.
            self._prefix_target(target_node.value)
        elif isinstance(target_node, ast.Starred):
            # This is a starred assignment like 'a, *b = ...'. Recurse on 'b'.
            self._prefix_target(target_node.value)
        
        return target_node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Handles simple assignments: a = b"""
        for target in node.targets:
            self._prefix_target(target)
        self.generic_visit(node) # Ensure we visit children on the right-hand side too
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """Handles annotated assignments: a: int = b"""
        self._prefix_target(node.target)
        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        """Handles augmented assignments: a += b"""
        self._prefix_target(node.target)
        self.generic_visit(node)
        return node

def add_underscore_to_lhs(code_string: str) -> str:
    """
    Parses a Python code string, adds an underscore to all variables on the
    left-hand side of assignments, and returns the modified code string.

    Args:
        code_string: A string containing one or more lines of Python code.

    Returns:
        The modified code string.
    
    Requires Python 3.9+ for ast.unparse().
    """
    try:
        # 1. Parse the string into an Abstract Syntax Tree
        tree = ast.parse(code_string)
        
        # 2. Instantiate our transformer and have it visit the tree
        transformer = UnderscoreLHS()
        new_tree = transformer.visit(tree)
        
        # 3. Add line numbers and other metadata back to the new tree
        ast.fix_missing_locations(new_tree)
        
        # 4. Unparse the modified tree back into a string
        return ast.unparse(new_tree)
    except (SyntaxError, ValueError) as e:
        print(f"Error processing code string: {e}")
        return code_string


# -- function parsers & code emitters --
def extract_vars(f: str, exclude: set) -> List[str]:
    """
    Extract variable names from a given string, excluding specified names.

    Parameters
    ----------
    f : str
        The input string from which to extract variable names.
    exclude : set
        A set of variable names to exclude from the result.

    Returns
    -------
    List[str]
        A list of variable names found in the input string, excluding the specified names.
    """
    pattern = r"\b[a-zA-Z_]\w*\b"
    all_variables = re.findall(pattern, f)
    filtered_variables = [var for var in all_variables if var not in exclude]
    return filtered_variables


def replace(input_string: str, replace_list: List[str]) -> str:
    """
    Replace occurrences of substrings in the input string with their 'self.' prefixed versions.

    Parameters
    ----------
    input_string : str
        The string in which to replace substrings.
    replace_list : list of str
        A list of substrings to be replaced.

    Returns
    -------
    str
        The modified string with specified substrings replaced by 'self.' prefixed versions.
    """
    for substring in replace_list:
        input_string = re.sub(rf"\b{substring}\b", f"self.{substring}", input_string)
    return input_string


def modify_operations(input_string: str) -> str:
    """
    Modify operations in the input string by prefixing PyTorch operations with 'torch.'.

    Parameters
    ----------
    input_string : str
        The input string containing code with function calls.

    Returns
    -------
    str
        The modified string with PyTorch operations prefixed by 'torch.'.
    """
    pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("

    def replacer(match):
        func_name = match.group(1)
        if (func_name in torch_operations):
            return f"torch.{func_name}("
        return match.group(0)

    modified_string = re.sub(pattern, replacer, input_string)
    return modified_string



def convert(deriv, state, states, assigned, use_pade_approx=False):
    """
    Convert a derivative expression into a form suitable for numerical integration.

    Parameters
    ----------
    deriv : str
        The derivative expression to be converted.
    state : str
        The state variable involved in the derivative.
    assigned : set
        A set of variables that are assigned values.
    use_pade_approx : bool, optional
        Whether to use Pade approximation for integration. Defaults to False.
    diffusion : Tuple[float, str], optional
        The diffusion to be added, if any. Defaults to None.
    model : object, optional
        The model object (subclass of Axon). Defaults to None.

    Returns
    -------
    str
        The modified derivative expression after integration and optional diffusion addition.
    """

    exclude = set([state]) | set(states) | set(assigned)
    v = extract_vars(deriv, exclude)
    f = integrate2c(deriv, "dt", v, use_pade_approx=use_pade_approx)
    if DEBUG:
        print(f)
    return add_underscore_to_lhs(modify_operations(replace(f, v)))


cnexp_template = """
def solve(self, dt, {states_and_assigned}):
    {solves}
    return {returns}
"""

def match_derivative_to_states(derivative, states):
    matched = {}
    for state in states:
        for d in derivative:
            s, _ = d.split("'")
            if s == state:
                matched[state] = d
                break
    return matched

def build_integration_func(states, assigned, derivative, method, pade=False):
    """
    Build the integration function for the states and assigned variables.
    """
    if method == 'cnexp':
        return build_cnexp(states, assigned, derivative, pade=pade)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def build_cnexp(states, assigned, derivative, pade=False):
    for state in states:
        if state in assigned:
            raise ValueError(f"State {state} cannot be assigned and used as a state variable.")
    states_and_assigned = ", ".join(set(states).union(assigned))
    solves = []
    returns = []

    if PADE.value == -1:
        use_pade_approx = pade
    else:
        use_pade_approx = bool(PADE)

    derivative = match_derivative_to_states(derivative, states)
    for state in states:
        solves.append(convert(derivative[state], state, states, assigned, use_pade_approx=use_pade_approx))
        returns.append(f"'{state}' : _{state}")
    solves = "\n    ".join(solves)
    returns = f"{{{', '.join(returns)}}}"
    f = cnexp_template.format(
        states_and_assigned=states_and_assigned,
        solves=solves,
        returns=returns
    )
    if DEBUG:
        print(f"Function:\n{f}")
    filename = "<solve_function>"
    code = compile(f, filename, "exec")
    exec(code)
    return locals()["solve"]


class State(Parameterized):

    _state_buffers = set()
    _state_buffers_declarations = []

    _state = set()
    _state_declarations = []

    _derivative = set()
    _derivative_declarations = []

    _assigned = set()
    _assigned_declarations = []

    has_q10 = False
    method = 'cnexp'

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called automatically whenever a class
        inherits from Parameterized.
        """
        # Call the parent's __init_subclass__ WITHOUT our custom kwargs,
        # as the base 'object' class does not accept them.
        super().__init_subclass__(**kwargs)

        # Start with a fresh dictionary for the new class's parameters.
        new_state = set()
        new_buffers = set()
        new_derivative = set()
        new_assigned = set()

        # Walk MRO in reverse to build up params from parent to child
        for base in reversed(cls.__mro__):
            # We look for a _params attribute defined directly on the base
            if '_state' in base.__dict__:
                new_state.update(base._state)
            if '_state_buffers' in base.__dict__:
                new_buffers.update(base._state_buffers)
            if '_derivative' in base.__dict__:
                new_derivative.update(base._derivative)
            if '_assigned' in base.__dict__:
                new_assigned.update(base._assigned)

        if State._state_declarations:
            for s_list in State._state_declarations:
                new_state.update(s_list)
            State._state_declarations = []
        
        if State._state_buffers_declarations:
            for b_list in State._state_buffers_declarations:
                new_buffers.update(b_list)
            State._state_buffers_declarations = []

        if State._derivative_declarations:
            for d_list in State._derivative_declarations:
                new_derivative.update(d_list)
            State._derivative_declarations = []

        if State._assigned_declarations:
            for a_list in State._assigned_declarations:
                new_assigned.update(a_list)
            State._assigned_declarations = []

        cls._state = list(new_state)
        cls._state_buffers = new_buffers
        cls._derivative = new_derivative
        cls._assigned = list(new_assigned)

    def __init__(
        self,
        celsius,
        diameters,
        key,
        shape,
        additional_parameters=None,
        **kwargs
    ):
        if not self._state:
            raise ValueError(f"State {self.__class__.__name__} has no state variables defined."
                              "Use State.STATE(<state vars>) in State implementation to define them.")
        super().__init__(shape, additional_parameters=additional_parameters, **kwargs)
        self._name = self.__class__.__name__
        self.key = key

        self.register_buffer('celsius', celsius)
        self.register_buffer('diam', diameters)

        for b in self._state_buffers:
            self.register_buffer(b, torch.tensor(0.0))

        pade = kwargs.get("pade", False)
        self.include_q10_in_comp_graph = kwargs.get("include_q10_in_comp_graph", False)

        ifunc = build_integration_func(
            self._state, self._assigned, self._derivative, self.method, pade
        )
        setattr(self, "solve", MethodType(ifunc, self))

    def populate_parameter_buffers(self):
        super().populate_parameter_buffers()
        if self.has_q10:
            if self.include_q10_in_comp_graph:
                self.q10 = self.calc_q10
            else:
                self.q10 = self.return_q10_cache
                self.register_buffer('q10_cache', self.calc_q10())

    @staticmethod
    def to_column(tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a 2D tensor to a column vector (2D tensor with one column).
        """
        return tensor.view(-1, 1)

    def from_column(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a column vector (2D tensor with one column) back to its original shape.
        """
        return tensor.view(*self.shape)

    def initialize(self, v):
        self.initial(v)
        return

    def return_q10_cache(self):
        return self.q10_cache

    def set(self, key: str, value):
        p = getattr(self, key)
        if isinstance(p, torch.Tensor):
            p.data = torch.as_tensor(value, dtype=p.data.dtype, device=p.device)

    @staticmethod
    def STATE(*args):
        State._state_declarations.append(args)

    @staticmethod
    def BUFFER(*args):
        State._state_buffers_declarations.append(args)

    @staticmethod
    def DERIVATIVE(*args):
        State._derivative_declarations.append(args)

    @staticmethod
    def ASSIGNED(*args):
        State._assigned_declarations.append(args)

    def breakpoint(self, v):
        return {}

    def advance(self, v, dt, states):
        return self.solve(dt, **self.breakpoint(v), **states)

    def initial(self, v):
        """
        Initial function to be called after the state is created.
        """
        pass
