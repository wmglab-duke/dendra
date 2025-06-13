from functools import partial
import inspect
import linecache
from typing import Dict, List

import torch
from torch.nn.utils import parametrize

from .txt import (
    template, 
    init_state_buffer_template, 
    init_state_buffers_coupled_template
)
from .ast import (
    factor_linear_in_x_from_codeblock,
    replace_v,
    factorize_linear_in_v,
)
from .utils import indent, load, get_function_body_as_str, get_function_as_str
from .compile_f import convert_func, multiply_return_value

from ..core import Mechanism, coupled
from ..state_compiler import compile_state, compile_coupled_state
from ..handler.defaults import valid_concentrations
from ..ops import *

from axonml.helpers import logger
from axonml.models.interfaces import AxonInterface
from axonml.models.parametric import to_param, positive, PositiveSoftplus

load = partial(load, cls=Mechanism)


default_f = """
    def {fname}(self, v):
{ret}
"""


generic_f = """
    def {fname}(self, model: AxonInterface):
{ret}
"""


def translate(mech, fname, template):
    f = getattr(mech, fname, None)
    if f:
        body = get_function_body_as_str(f)
    else:
        return ""
    return template.format(fname=fname, ret=body)


def translate_if_exists(mech, fname):
    f = getattr(mech, fname, None)
    if f:
        return get_function_as_str(f)
    return ""


mech_inf_template = """
def {state}_inf(self, v):
    return torch.tensor(0.0, device=v.device, dtype=v.dtype)
"""


def define_coupled_infs(mechanism, states):
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
            lhs.append(f"self.{state_name}")
            rhs.append(f"self.{state_name}")
    lhs = ", ".join(lhs)
    rhs = ", ".join(rhs)
    return template.format(name=name, lhs=lhs, rhs=rhs)


def mask_to_index(mask_out, mask_in, n_comps):
    if mask_out is None and mask_in is None:
        return None
    if mask_out is not None and mask_in is not None:
        raise ValueError("Only one of mask_out or mask_in can be specified.")
    if mask_out is not None:
        return [i for i in range(n_comps) if i not in mask_out]
    if mask_in is not None:
        return [i for i in range(n_comps) if i in mask_in]
    return None


class MechCompiler:
    def __init__(self, DEBUG=0, DETECT_ANOMALIES=0, PADE=-1):
        self.DEBUG = DEBUG
        self.PADE = PADE
        self.DETECT_ANOMALIES = DETECT_ANOMALIES

    @staticmethod
    def mask(mask_out, mask_in):
        if mask_out is None and mask_in is None:
            return ""
        ret = []
        if mask_out is not None:
            ret.append(
                f"mask_out = torch.ones(1, n_comps)\nmask_out[:, {mask_out}] = 0"
            )
        if mask_in is not None:
            ret.append(
                f"mask_in = torch.zeros(1, n_comps)\nmask_in[:, {mask_in}] = 1"
            )
        if mask_out is not None and mask_in is not None:
            ret.append("mask = mask_out * mask_in")
        elif mask_out is not None:
            ret.append("mask = mask_out")
        else:
            ret.append("mask = mask_in")
        ret.append("self.register_buffer('mask', mask)")
        # ret.append("self.register_buffer('idx', mask_)")
        return indent("\n".join(ret), 2)
    
    @staticmethod
    def detach(to_detach):
        assignments = []
        for k in to_detach:
            assignments.append(f"self.{k}.detach_()")
        return indent("\n".join(assignments), 2)
    
    @staticmethod
    def buffers(names, mask=None):
        assignments = []
        for n in names:
            assignments.append(
                f"self.register_buffer('{n}', torch.zeros(shape))"
            )
        return indent("\n".join(assignments), 2)
    
    @staticmethod
    def current_buffers(names):
        assignments = []
        for n in names:
            assignments.append(
                f"self.register_buffer('{n}_', torch.zeros(shape))"
            )
        return indent("\n".join(assignments), 2)
    
    @staticmethod
    def assigned(names):
        assignments = []
        for n in names:
            assignments.append(f"self.register_buffer('{n}', torch.tensor(0.0))")
        return indent("\n".join(assignments), 2)
    
    def advance(self, states, mask=None):
        assignments = []
        #if mask is not None:
        #    assignments.append(f"v = v.index_select(-1, self.idx)")
        for k in states:
            if k.coupled:
                assignments.append(coupled_assignment(k))
            else:
                name = k._name
                if name in valid_concentrations():
                    assignments.append(
                        f"self.{name} = self.DE['{name}'].advance(self.{name}, v, dt)"
                    )
                else:
                    assignments.append(
                        f"self.{name} = self.DE['{name}'].advance(self.{name}, v, dt)"
                    )
        if self.DETECT_ANOMALIES:
            for k in states:
                if k.coupled:
                    for name in k._state_names:
                        assignments.append(
                            f"assert torch.all(torch.isfinite(self.{name})), 'Anomaly detected in {name}'"
                        )
                else:
                    name = k._name
                    assignments.append(
                        f"assert torch.all(torch.isfinite(self.{name})), 'Anomaly detected in {name}'"
                    )
        return indent("\n".join(assignments), 2)
    
    @staticmethod
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
        return indent("\n".join(assignments), 2)
    
    @staticmethod
    def init_distribution_buffers(distributions):
        assignments = []
        for k in distributions:
            assignments.append(
                f"self.{k} = self.{k}_d._sample(self.{k})"
            )
        return indent("\n".join(assignments), 2)
    
    @staticmethod
    def implicit_buffers(names):
        return ""

    @staticmethod
    def gtot(currents, mechanism, mask):
        return "    def gtot(self): return 0.0", False
    
    @staticmethod
    def irev(currents, mask):
        return ""
    
    @staticmethod
    def current_equations(currents, mechanism, range_vars, mask):
        assignments = []
        unfactorable = None
        divide_by_two = {}
        for k in currents:
            assign = k in range_vars
            code = convert_func(getattr(mechanism, k), assign)
            if mask:
                code = multiply_return_value(code, "self.mask")
            assignments.append(code)
        return indent("\n".join(assignments), 1), unfactorable, divide_by_two

    def compile(
            self, 
            mechanism, 
            model, 
            ic=None, 
            mask_out=None, 
            mask_in=None, 
            **kwargs
        ):
        temp = model.temp
        diameters = model.diam
        shape = model.shape
        n_ax = model.n_ax
        n_comps = model.n_comp
        pade = None if self.PADE < 0 else bool(self.PADE)

        states = mechanism._states

        params = load(mechanism, "_params")
        params, distributions = parse_params_distributions(params, kwargs)
        init = load(mechanism, "_init")
        currents = load(mechanism, "_currents")
        range_vars = load(mechanism, "_range")
        assigned = load(mechanism, "_assigned")

        read_ion = load(mechanism, "_read_ion")
        write_ion = load(mechanism, "_write_ion")
        write_ion_c = load(mechanism, "_write_ion_c")

        # identify currents
        current_eqs = []
        for _, v in currents.items():
            current_eqs.extend(v)
        for _, v in write_ion.items():
            current_eqs.extend(v)

        # compile states
        states_compiled = []
        for s in states:
            if coupled(s):
                states_compiled.append(compile_coupled_state(s, model, pade=pade, **kwargs))
            else:
                states_compiled.append(compile_state(s, model, pade=pade, **kwargs))

        # buffers
        state_names = []
        for s in states_compiled:
            if not s.coupled:
                state_names.append(s._name)
            else:
                state_names.extend(s._state_names)
        
        current_names = []
        all_current_names = []
        for k in current_eqs:
            all_current_names.append(k)
            if k in range_vars:
                current_names.append(k)

        # detach
        to_detach = []
        for k in states_compiled:
            if not k.coupled:
                to_detach.append(k._name)
            else:
                to_detach.extend(k._state_names)
        for k in assigned:
            to_detach.append(k)
        for _, v in read_ion.items():
            for v_ in v:
                to_detach.append(v_)

        mask = (mask_out is not None) or (mask_in is not None)

        current_eqs_str, unfactorable, divide_by_two = self.current_equations(
            current_eqs, mechanism, range_vars, mask
        )
        gtot, has_gtot = self.gtot(current_eqs, mechanism, mask)

        compiled_str = template.format(
            mech                        = mechanism.__name__,
            mask                        = self.mask(mask_out, mask_in),
            state_buffers               = self.buffers(state_names),
            distribution_buffers        = self.buffers(distributions.keys()),
            current_buffers             = self.current_buffers(current_names),
            implicit_buffers            = self.implicit_buffers(all_current_names),
            assigned                    = self.assigned(assigned),
            detach                      = self.detach(to_detach),
            init_state_buffers          = self.init_state_buffers(states_compiled),
            init_distribution_buffers   = self.init_distribution_buffers(distributions),
            advance                     = self.advance(states_compiled, mask),
            initial_f                   = translate(mechanism, "initial", default_f),
            breakpoint_f                = translate(mechanism, "breakpoint", default_f),
            generic_f                   = translate(mechanism, "generic", generic_f),
            coupled_infs                = define_coupled_infs(mechanism, states_compiled),
            net_receive_f               = translate_if_exists(mechanism, "net_receive"),
            irev                        = self.irev(current_eqs, mask),
            gtot                        = gtot,
            current_equations           = current_eqs_str,
        )

        if self.DEBUG >= 2:
            print(compiled_str)

        # compile the mechanism
        filename = f"<{mechanism.__name__}>_template"
        code = compile(compiled_str, filename, "exec")
        exec(code)

        lines = [line + "\n" for line in compiled_str.splitlines()]
        linecache.cache[filename] = (len(lines), None, lines, filename)
        name = mechanism.__name__

        m = locals()[name](
            temp,
            diameters,
            shape,
            n_ax,
            n_comps,
            name,
            params,
            distributions,
            read_ion,
            write_ion_c,
            states_compiled,
            init,
            model,
            ic=ic,
        )

        return m, unfactorable, has_gtot, divide_by_two


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


df_equation_template = """
def {k}(self, v):
    gt = {gtot}
    i = gt * (0.5 * v - {irev})
    self.gtot_{k} = gt
    return i
"""


df_itot_template = """
def {k}_tot(self, v):
    i = self.gtot_{k} * (v - {irev})
    {assign_to_buffer}
    return i
"""


def build_df_equation(current, gtot, irev):
    return df_equation_template.format(
        k=current,
        gtot=gtot,
        irev=irev,
    )


def build_df_itot(current, irev, assign):
    if assign:
        assign_to_buffer = f"self.{current}_ = i"
    else:
        assign_to_buffer = ""
    return df_itot_template.format(
        k=current,
        irev=irev,
        assign_to_buffer=assign_to_buffer,
    )


class DF_Compiler(MechCompiler):
    def __init__(self, DEBUG=0, DETECT_ANOMALIES=0, PADE=-1):
        super().__init__(DEBUG, DETECT_ANOMALIES, PADE)

    @staticmethod
    def _implicit_buffers(names):
        assignments = []
        for n in names:
            assignments.append(
                f"self.register_buffer('gtot_{n}', torch.zeros(shape))"
            )
        return indent("\n".join(assignments), 2)

    @staticmethod
    def gtot(currents, mechanism, mask):
        mult = " * self.mask" if mask else ""
        if hasattr(mechanism, "conductance"):
            return f"    def gtot(self, v): return 0.5 * self.conductance(v) {mult}", True
        
        assignments = []
        for k in currents:
            code_blocks = get_function_body_as_str(getattr(mechanism, k))
            try:
                _, b = factor_linear_in_x_from_codeblock(replace_v(code_blocks))
                assignments.append(b)
            except:
                pass
        
        has_gtot = True
        if not assignments:
            has_gtot = False
            return "    def gtot(self, v): return 0.0", has_gtot
        
        s = " + ".join(assignments)
        return f"    def gtot(self, v): return ({s}) {mult}", has_gtot
    
    def _gtot(self, currents, mechanism, mask):
        mult = " * self.mask" if mask else ""
        
        assignments = []
        for k in currents:
            if k in self.unfactorable:
                continue
            if hasattr(mechanism, f"conductance_{k}"):
                assignments.append(f"self.conductance_{k}(v)")
            else:
                assignments.append(f"self.gtot_{k}")
        
        has_gtot = True
        if not assignments:
            has_gtot = False
            return "    def gtot(self, v): return 0.0", has_gtot
        
        s = " + ".join(assignments)
        return f"    def gtot(self, v): return 0.5 * ({s}) {mult}", has_gtot
    
    def _current_equations(self, currents, mechanism, range_vars, mask):
        assignments = []
        unfactorable = []
        divide_by_two = {}
        for k in currents:
            assign = k in range_vars
            try:
                gtot, irev = factorize_linear_in_v(mechanism, method=k)
                code = build_df_equation(k, gtot, irev)
                if mask:
                    code = multiply_return_value(code, "self.mask")
                assignments.append(code)
                code = build_df_itot(k, irev, assign)
                if mask:
                    code = multiply_return_value(code, "self.mask")
                assignments.append(code)
            except:
                logger.warning(
                    f"Could not factorize {k} in {mechanism.__name__}."
                )
                code = convert_func(getattr(mechanism, k), assign)
                if mask:
                    code = multiply_return_value(code, "self.mask")
                assignments.append(code)

                code_block = get_function_body_as_str(getattr(mechanism, k))
                if mask:
                    code_block_tot = multiply_return_value(code_block, "self.mask")
                else:
                    code_block_tot = code_block
                assignments.append(current_tot_template.format(k=k, body=code_block_tot))

                if not hasattr(mechanism, f"conductance_{k}"):
                    logger.warning(
                        f"Could not find conductance function for {k} in {mechanism.__name__}."
                    )
                    unfactorable.append(k)
                else:
                    code = convert_func(getattr(mechanism, f"conductance_{k}"), False)
                    if mask:
                        code = multiply_return_value(code, "self.mask")
                    assignments.append(code)
        self.unfactorable = unfactorable
        return indent("\n".join(assignments), 1), unfactorable, divide_by_two
    
    @staticmethod
    def current_equations(currents, mechanism, range_vars, mask):
        assignments = []
        unfactorable = []
        divide_by_two = {}

        for k in currents:
            assign = k in range_vars
            code_block = get_function_body_as_str(getattr(mechanism, k))
            if mask:
                code_block_tot = multiply_return_value(code_block, "self.mask")
            else:
                code_block_tot = code_block
            assignments.append(current_tot_template.format(k=k, body=code_block_tot))
            try:
                i, _ = factor_linear_in_x_from_codeblock(replace_v(code_block))
                if assign:
                    code = current_eq_template_assign.format(k=k, v=i)
                    if mask:
                        code = multiply_return_value(code, "self.mask")
                    assignments.append(code)
                else:
                    code = current_eq_template.format(k=k, v=i)
                    if mask:
                        code = multiply_return_value(code, "self.mask")
                    assignments.append(code)
                divide_by_two[k] = False
            except:
                logger.warning(
                    f"Could not factorize {k} in {mechanism.__name__}; looking instead for user-defined functions."
                )
                if not hasattr(mechanism, "conductance"):
                    logger.warning(
                        f"Could not find conductance function in {mechanism.__name__}."
                    )
                    code = convert_func(getattr(mechanism, k), assign)
                    divide_by_two[k] = False
                elif hasattr(mechanism, f"{k}_df"):
                    logger.info(
                        f"Conductance function found in {mechanism.__name__}. Using {k}_df function."
                    )
                    code = convert_func(getattr(mechanism, f"{k}_df"), assign, k)
                    divide_by_two[k] = False
                else:
                    logger.info(
                        f"Conductance function found in {mechanism.__name__}. No {k}_df function found, using {k}."
                    )
                    code = convert_func(getattr(mechanism, k), assign)
                    divide_by_two[k] = True
                if mask:
                    code = multiply_return_value(code, "self.mask")
                assignments.append(code)
                unfactorable.append(k)

        if hasattr(mechanism, "conductance"):
            code = convert_func(mechanism.conductance, False)
            if mask:
                code = multiply_return_value(code, "self.mask")
            assignments.append(code)

        return indent("\n".join(assignments), 1), unfactorable, divide_by_two


implicit_equation_template = """
def {k}(self, v):
    gtot_ = {gtot}
    irev = {irev}
    self.gtot_{k} = gtot_
    i = gtot_ * (v - irev)
    {assign_to_buffer}
    return i
"""


mask_v_template = "v = v.index_select(-1, self.idx)"


def build_mask_v(mask):
    if mask is None:
        return ""
    return mask_v_template.format(mask=mask)


def gtot_irev_i_z(mask):
    if mask is None:
        return "", "", ""
    return (
        "gtot_z = torch.zeros_like(v)", 
        "irev_z = torch.zeros_like(v)", 
        "i_z = torch.zeros_like(v)"
    )


def gtot_irev_build(current, mask):
    if mask is None:
        return f"gtot_", f"irev_"
    else:
        return (
            f"self.rebuild(gtot_.expand_as(v))", 
            f"self.rebuild(irev_.expand_as(v))"
        )

def i_build(mask):
    if mask is None:
        return "i = i_"
    else:
        return "i = self.rebuild(i_)"


def build_implicit_equation(current, gtot, irev, assign):
    if assign:
        assign_to_buffer = f"self.{current}_ = i"
    else:
        assign_to_buffer = ""
    return implicit_equation_template.format(
        k=current,
        gtot=gtot,
        irev=irev,
        assign_to_buffer=assign_to_buffer,
    )


class ImplicitCompiler(MechCompiler):
    def __init__(self, DEBUG=0, DETECT_ANOMALIES=0, PADE=-1):
        super().__init__(DEBUG, DETECT_ANOMALIES, PADE)
    
    def gtot(self, currents, mechanism, mask):
        mult = " * self.mask" if mask else ""
        factorable = []
        
        assignments = []
        for k in currents:
            if k in self.unfactorable:
                continue
            if hasattr(mechanism, f"conductance_{k}"):
                assignments.append(f"self.conductance_{k}(v)")
            else:
                assignments.append(f"self.gtot_{k}")
                factorable.append(k)

        self.factorable = factorable
        
        has_gtot = True
        if not assignments:
            has_gtot = False
            return "    def gtot(self, v): return 0.0", has_gtot
        
        s = " + ".join(assignments)
        return f"    def gtot(self, v): return ({s}) {mult}", has_gtot
    
    def current_equations(self, currents, mechanism, range_vars, mask):
        assignments = []
        unfactorable = []
        divide_by_two = {}
        for k in currents:
            assign = k in range_vars
            try:
                gtot, irev = factorize_linear_in_v(mechanism, method=k)
                code = build_implicit_equation(k, gtot, irev, assign)
                if mask:
                    code = multiply_return_value(code, "self.mask")
                assignments.append(code)
            except:
                logger.warning(
                    f"Could not factorize {k} in {mechanism.__name__}."
                )
                code = convert_func(getattr(mechanism, k), assign)
                if mask:
                    code = multiply_return_value(code, "self.mask")
                assignments.append(code)

                if not hasattr(mechanism, f"conductance_{k}"):
                    logger.warning(
                        f"Could not find conductance function for {k} in {mechanism.__name__}."
                    )
                    unfactorable.append(k)
                else:
                    code = convert_func(getattr(mechanism, f"conductance_{k}"), False)
                    if mask:
                        code = multiply_return_value(code, "self.mask")
                    assignments.append(code)
        self.unfactorable = unfactorable
        return indent("\n".join(assignments), 1), unfactorable, divide_by_two
    
    @staticmethod
    def implicit_buffers(names):
        assignments = []
        for n in names:
            assignments.append(
                f"self.register_buffer('gtot_{n}', torch.zeros(shape))"
            )
            assignments.append(
                f"self.register_buffer('irev_{n}', torch.zeros(shape))"
            )
        return indent("\n".join(assignments), 2)
    
    def irev(self, currents, mask):
        mult = " * self.mask" if mask else ""
        assignments = []
        for k in currents:
            if k not in self.unfactorable:
                assignments.append(f"self.irev_{k}")
        total = " + ".join(assignments)
        return f"    def irev(self): return {total} {mult}"

    def detach(self, to_detach):
        for k in self.factorable:
            to_detach.append(f"gtot_{k}")
            to_detach.append(f"irev_{k}")
        return super().detach(to_detach)
        
