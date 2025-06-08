from functools import partial
from typing import List, Tuple

import inspect
import linecache
import re
import math

import torch

from .ode import integrate2c

from .core import State
from .ops import *
from .utils import load, randomword, indent, get_function_body_as_str
from axonml.helpers import DEBUG

# -- used in generated code --
from axonml.models.math.diffusion import diffuse_step_neumann_dct1
from axonml.models.parametric import to_param

from axonml.helpers import interp1d


# PyTorch operations
torch_operations = set(dir(torch))


# utility functions
load = partial(load, cls=State)


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


# -- add diffusion --


def diffusion_expr(state: str, D: float, method: str, L: float, n: int) -> str:
    """
    Generate a diffusion expression string for a given state using the specified method.

    Parameters
    ----------
    state : str
        The state variable to be diffused.
    D : float
        The diffusion coefficient.
    method : str
        The method to be used for diffusion. Supported methods are 'strang'.
    L : float
        The length scale for the diffusion process.
    n : int
        The number of nodes for the diffusion process.

    Returns
    -------
    str
        A string representing the diffusion step expression.
    """
    dt_str = "dt"
    if method == "strang":
        dt_str = "dt / 2"
    return f"{state} = diffuse_step_neumann_dct1({state}, {dt_str}, {D}, {L}, {n})"


def add_diffusion(f: str, state: str, diffusion: Tuple[float, str], model) -> str:
    """
    Add diffusion step(s) to the given code string for a specified state variable.

    Parameters
    ----------
    f : str
        The input code string to which the diffusion step will be added.
    state : str
        The state variable to be diffused.
    diffusion : Tuple[float, str]
        A tuple containing the diffusion coefficient and method (e.g., 'strang').
    model : object
        The model object (subclass of Axon) containing the domain length and number of nodes.

    Returns
    -------
    str
        The modified code string with the diffusion step(s) added.
    """
    L = model.dx * (model.n_comp - 1)
    D, method = diffusion
    diff = diffusion_expr(state, D, method, L, model.n_comp)
    f = f"{diff} ; {f}"
    if method == "strang":
        f = f"{f} ; {diff}"
    return f


def convert(deriv, state, assigned, use_pade_approx=False, diffusion=None, model=None):
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
    v = extract_vars(deriv, set([state]) | assigned)
    f = integrate2c(deriv, "dt", v, use_pade_approx=use_pade_approx)
    if diffusion is not None:
        f = add_diffusion(f, state, diffusion, model)
    if DEBUG:
        print(f)
    return modify_operations(replace(f, v))


# ------------------------------
# -- code templates & helpers --
# ------------------------------

template = """
class _state_{name}(torch.nn.Module):
    def __init__(self, temp, diameters, is_q10: bool, name: str, buffers: List[str], params, **kwargs):
        super().__init__()

        self.coupled = False
        self.instantiate_parameters(params, **kwargs)

        self.register_buffer("diam", diameters)
        self.register_buffer("temp", torch.as_tensor(temp))
        self._name = name

        self.is_q10 = is_q10
        if self.is_q10:
            self.register_buffer("q10_cache", torch.as_tensor(self.calc_q10()))

        for b in buffers:
            self.register_buffer(b, torch.tensor(0.0))

    def initialize(self, v):
        self.initial(v)
        return

    def eval(self):
        if self.is_q10:
            self.q10_cache = self.calc_q10()
            self.q10 = self.return_q10_cache
        return super().eval()

    def train(self, mode=True):
        if self.is_q10:
            self.q10 = self.calc_q10
        return super().train(mode)

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

{initial_f}

{helpers}
"""


template_coupled = """
class _state_{name}(torch.nn.Module):
    def __init__(self, temp, diameters, is_q10: bool, name: str, state_names: List[str], buffers: List[str], params, **kwargs):
        super().__init__()

        self.coupled = True

        self.instantiate_parameters(params, **kwargs)

        self.register_buffer("diam", diameters)
        self.register_buffer("temp", torch.as_tensor(temp))

        self._name = name
        self._state_names = state_names

        self.is_q10 = is_q10
        if self.is_q10:
            self.register_buffer("q10_cache", self.calc_q10())

        for b in buffers:
            self.register_buffer(b, torch.tensor(0.0))

    def initialize(self, v):
        self.initial(v)
        return

    def eval(self):
        if self.is_q10:
            self.q10_cache = self.calc_q10()
            self.q10 = self.return_q10_cache
        return super().eval()

    def train(self, mode=True):
        if self.is_q10:
            self.q10 = self.calc_q10
        return super().train(mode)

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

    def advance(self, {old_states}, v, dt):
        return self.integrate({old_states}, dt {breakpoint_int})

    def integrate(self, {old_states}, dt {integrate_args}):
{integrate_fs}
        return {states}

{breakpoint_f}

{initial_f}

{calc_q10_f}

{helpers}
"""

default_f = """
    def {fname}(self, v):
{ret}
"""

breakpoint_str = """
    def breakpoint(self, v):
{body}
{ret}
"""

initial_str = """
    def initial(self, v):
{body}
        return
"""


def assigned_str_f(assigned):
    assignments = []
    for a in assigned:
        assignments.append(f"self.register_buffer('{a}', torch.tensor(0.0))")
    return "\n".join(assignments)


def translate_breakpoint(state: State, assigned):
    """
    Translates the `breakpoint` function from a State object into a formatted string.

    Parameters
    ----------
    state : State
        The State object containing the breakpoint function.
    assigned : list
        List of assigned variables to be returned by the breakpoint function.

    Returns
    -------
    str
        A formatted string representing the breakpoint function, or an empty string
        if the state object does not have a breakpoint function.
    """
    if not assigned:
        return ""
    f = getattr(state, "breakpoint", None)
    if not f:
        raise ValueError(f"Must specify breakpoint function to compute assigned variables: {assigned}.")
    body = get_function_body_as_str(f)
    return_names = ", ".join(assigned)
    return breakpoint_str.format(
        body=body, ret=indent("return " + f"({return_names},)", 2)
    )


def translate_initial(state: State):
    """
    Translates the `initial` function of a given State object into a formatted string.

    Parameters
    ----------
    state : State
        The State object containing the initial state to be translated.

    Returns
    -------
    str
        A formatted string representing the initial function. If the State object
        does not have an 'initial' function, an empty body is returned.
    """
    f = getattr(state, "initial", None)
    if not f:
        return initial_str.format(body="")
    body = get_function_body_as_str(f)
    return initial_str.format(body=body)


def translate_f(state: State, fname: str, default=None):
    """
    Translates a function from the given state object into a string representation.

    Parameters
    ----------
    state : State
        The state object containing the function to be translated.
    fname : str
        The name of the function to be translated.
    default : str, optional
        The default string to use if the function is not found.

    Returns
    -------
    str
        The string representation of the function or the default string if the function is not found.
    """
    f = getattr(state, fname, None)
    if f:
        body = get_function_body_as_str(f)
    else:
        if default is None:
            default = "return"
        body = indent(default, 2)
    return default_f.format(fname=fname, ret=body)


def translate_q10(s: State, is_q10):
    """
    Translates the Q10 calculation function to its source code if applicable.

    Parameters
    ----------
    s : State
        State object that may contain a method named 'calc_q10'.
    is_q10 : bool
        A flag indicating whether Q10 calculation is required.

    Returns
    -------
    str
        The source code of the 'calc_q10' function if 'is_q10' is True,
        otherwise an empty string.

    Raises
    ------
    ValueError
        If 'is_q10' is True and the 'calc_q10' function is not found.
    """
    if not is_q10:
        return ""
    q10_f = getattr(s, "calc_q10", None)
    if q10_f is None:
        raise ValueError("Must specify calc_q10 function")
    return inspect.getsource(q10_f)


def collect_helper_functions(state: State):
    """
    Collects the source code of helper functions from the given state object.

    Parameters
    ----------
    state : State
        The state object containing the functions to be collected.

    Returns
    -------
    str
        A string containing the source code of the collected helper functions,
        concatenated together with newline characters.
    """
    functions = inspect.getmembers(state, predicate=inspect.isfunction)
    ret = []
    for fname, f in functions:
        if fname not in ["breakpoint", "inf", "calc_q10", "initial"]:
            ret.append(inspect.getsource(f))
    return "\n".join(ret)


def breakpoint_args(s: State):
    """
    Retrieves the breakpoint attribute from the given State object and formats it.

    Parameters
    ----------
    s : State
        The State object from which to retrieve the breakpoint attribute.

    Returns
    -------
    str
        A formatted string containing the breakpoint attribute if it exists,
        otherwise an empty string.
    """
    f = getattr(s, "breakpoint", None)
    if not f:
        return ""
    return ", *self.breakpoint(v)"


def integrate_args(assigned):
    """
    Formats the assigned variables into a string for integration arguments.

    Parameters
    ----------
    assigned : list
        A list of assigned variables.

    Returns
    -------
    str
        A formatted string of assigned variables, or an empty string if none exist.
    """
    if not assigned:
        return ""
    return ", " + ", ".join(assigned)


def extract_state(input_string):
    """
    Extracts the state variable name from an input string.

    Parameters
    ----------
    input_string : str
        The input string containing the state assignment.

    Returns
    -------
    str
        The extracted state variable name if a match is found, otherwise None.
    """
    match = re.match(r"([a-zA-Z_][a-zA-Z_0-9]*)'\s*=\s*.*", input_string)
    if match:
        return match.group(1)
    return None


def construct_integrate_fs_for_coupled(
    derivative, states, assigned, use_pade_approx=False
):
    """
    Constructs integration formulas for coupled differential equations.

    Parameters
    ----------
    derivative : list of str
        List of derivative expressions to be integrated.
    states : set of str
        Set of state variables involved in the differential equations.
    assigned : set of str
        Set of assigned (non-state) variables used in the equations.
    use_pade_approx : bool, optional
        Whether to use Padé approximation for integration. Defaults to False.

    Returns
    -------
    str
        A string representation of the integration formulas, with each formula
        on a new line. The state variables in the formulas are suffixed with '_old'.
    """
    output = []
    for d in derivative:
        state = extract_state(d)
        f_str = convert(d, state, assigned | states, use_pade_approx=use_pade_approx)
        f_str = add_suffix_to_variables(f_str, states, "_old")
        output.append(f_str)
    return "\n".join(output)


def add_suffix_to_variables(code: str, match_strings: set, suffix: str) -> str:
    """
    Add a suffix to variables that match any of a given set of strings
    on the right-hand side of an assignment statement.

    Parameters
    ----------
    code : str
        The input Python code as a string.
    match_strings : set
        A set of strings to match variables.
    suffix : str
        The suffix to add to matching variables.

    Returns
    -------
    str
        The modified Python code with the specified suffix added to matching variables.
    """
    # Define a regex to match assignment statements
    assignment_pattern = re.compile(r"(?P<lhs>.+?)=(?P<rhs>.+)")

    def modify_rhs(rhs: str) -> str:
        """Modify the RHS by adding suffix to matching variables."""
        # Split the RHS into tokens (may include variables, operators, etc.)
        tokens = re.split(r"(\W+)", rhs)

        # Add suffix to variables that match the target strings
        modified_tokens = [
            f"{token}{suffix}" if token in match_strings else token for token in tokens
        ]
        return "".join(modified_tokens)

    # Process each line of code
    modified_code = []
    for line in code.splitlines():
        match = assignment_pattern.match(line)
        if match:
            lhs = match.group("lhs").strip()
            rhs = match.group("rhs").strip()
            modified_rhs = modify_rhs(rhs)
            modified_code.append(f"{lhs} = {modified_rhs}")
        else:
            modified_code.append(line)  # Leave non-assignment lines unchanged

    return "\n".join(modified_code)


def compile_state(s: State, model, pade=None, **kwargs) -> torch.nn.Module:
    """
    Compile a state object into a PyTorch module.

    Parameters
    ----------
    s : State
        The state object to be compiled.
    model : Axon
        The model containing temperature and diameter information.
    pade : optional
        Padé approximation to be used. Defaults to None.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    torch.nn.Module
        The compiled state object as a PyTorch module.

    Raises
    ------
    ValueError
        If the derivative function is not specified.

    Notes
    -----
    - This function generates a string representation of the state object, including
      its parameters, assigned variables, and derivative function.
    - The generated string is compiled into executable code, which is then executed
      to create the PyTorch module.
    """

    temp = model.temp
    diameters = model.diam

    name = s.__name__
    params = load(s, "_params")
    assigned = load(s, "_assigned")
    assigned_list = list(assigned)
    is_q10 = load(s, "is_q10")
    derivative = load(s, "_derivative")
    buffers = load(s, "_buffers")
    diffusion = load(s, "_diffusion")

    if derivative is None:
        raise ValueError("Must specify derivative function")

    assigned_str = assigned_str_f(assigned)
    assigned_str = indent(assigned_str, 2)

    pade_approx = pade if pade is not None else derivative[1]

    integrate_f = convert(
        derivative[0],
        name,
        assigned,
        use_pade_approx=pade_approx,
        diffusion=diffusion,
        model=model,
    )

    forward_str = template.format(
        name=name,
        state=name,
        breakpoint_int=breakpoint_args(s),
        integrate_f=integrate_f,
        breakpoint_f=translate_breakpoint(s, assigned_list),
        integrate_args=integrate_args(assigned_list),
        inf_f=translate_f(
            s, "inf", "return self.alpha(v) / (self.alpha(v) + self.beta(v))"
        ),
        calc_q10_f=translate_q10(s, is_q10),
        helpers=collect_helper_functions(s),
        initial_f=translate_initial(s),
    )

    if DEBUG >= 3:
        print(forward_str)

    filename = f"<{name}_{randomword(5)}_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    m = locals()[f"_state_{name}"](
        temp, diameters, is_q10, name, buffers, params, **kwargs
    )

    return m


def compile_coupled_state(s: State, model, pade=None, **kwargs) -> torch.nn.Module:
    """
    Compile a coupled state object into a PyTorch module.

    Parameters
    ----------
    s : State
        The state object containing state information.
    model : Axon
        The model object containing temperature and diameter information.
    pade : optional
        Padé approximation to be used. Defaults to None.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    torch.nn.Module
        An instance of the compiled coupled state as a PyTorch module.

    Raises
    ------
    ValueError
        If the derivative function is not specified.

    Notes
    -----
    - This function generates a string representation of the coupled state object,
      including its parameters, assigned variables, and derivative functions.
    - The generated string is compiled into executable code, which is then executed
      to create the PyTorch module.
    """

    temp = model.temp
    diameters = model.diam

    name = s.__name__
    state_names = s._states
    state_name_list = list(state_names)

    params = load(s, "_params")
    assigned = load(s, "_assigned")
    assigned_list = list(assigned)
    is_q10 = load(s, "is_q10")
    derivative = load(s, "_derivative")
    buffers = load(s, "_buffers")

    old_states = ", ".join([f"{s}_old" for s in state_name_list])

    if derivative is None:
        raise ValueError("Must specify derivative function")

    assigned_str = assigned_str_f(assigned)
    assigned_str = indent(assigned_str, 2)

    pade_approx = pade if pade is not None else derivative[1]

    integrate_fs = construct_integrate_fs_for_coupled(
        derivative[0], state_names, assigned, use_pade_approx=pade_approx
    )

    forward_str = template_coupled.format(
        name=name,
        states=", ".join(state_name_list),
        old_states=old_states,
        breakpoint_int=breakpoint_args(s),
        integrate_fs=indent(integrate_fs, 2),
        breakpoint_f=translate_breakpoint(s, assigned_list),
        integrate_args=integrate_args(assigned_list),
        calc_q10_f=translate_q10(s, is_q10),
        initial_f=translate_initial(s),
        helpers=collect_helper_functions(s),
    )

    if DEBUG >= 3:
        print(DEBUG.value, forward_str)

    filename = f"<{randomword(7)}_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    m = locals()[f"_state_{name}"](
        temp,
        diameters,
        is_q10,
        name,
        state_name_list,
        buffers,
        params,
        **kwargs,
    )
    return m
