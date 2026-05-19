from dendra.helpers import DEBUG, logger

from .compilers.ast import factorize_linear_in_v

implicit_equation_template = """
def {k}(self, v):
    gtot_ = {gtot}
    irev = {irev}
    i = gtot_ * (v - irev)
    {assign_to_buffer}
    return i, gtot_
"""


implicit_equation_unfactorable_template = """
def {k}(self, v):
    i = self.{k}(v)
    {assign_to_buffer}
    return i, 0.0
"""


numerical_template = """
def {k}(self, v):
    i = self.{k}(v)
    i_d = self.{k}(v + 1e-3)
    {assign_to_buffer}
    return i, (i_d - i) / (1e-3)
"""


current_tot_template = """
def {k}_tot(self, v):
{body}
"""


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


def build_numerical_equation(current, assign):
    if assign:
        assign_to_buffer = f"self.{current}_ = i"
    else:
        assign_to_buffer = ""
    return numerical_template.format(
        k=current,
        assign_to_buffer=assign_to_buffer,
    )


def build_unfactorable_equation(current, assign):
    if assign:
        assign_to_buffer = f"self.{current}_ = i"
    else:
        assign_to_buffer = ""
    return implicit_equation_unfactorable_template.format(
        k=current,
        assign_to_buffer=assign_to_buffer,
    )


def build_current_eq(mechanism, k, assign=False):
    if hasattr(mechanism, f"{k}_with_conductance"):
        return getattr(mechanism, f"{k}_with_conductance"), True
    if k in mechanism._numerical:
        code = build_numerical_equation(k, assign)
        factorable = True
        if DEBUG > 0:
            logger.info(f"Generated code for {k}:\n{code}")
        filename = "<solve_function>"
        code = compile(code, filename, "exec")
        exec(code)
        return locals()[k], factorable
    try:
        gtot, irev = factorize_linear_in_v(mechanism.__class__, method=k)
        code = build_implicit_equation(k, gtot, irev, assign)
        factorable = True
    except Exception:
        if k not in mechanism._explicit:
            code = build_numerical_equation(k, assign)
            factorable = True
        else:
            code = build_unfactorable_equation(k, assign)
            factorable = False
    if DEBUG > 0:
        logger.info(f"Generated code for {k}:\n{code}")
    filename = "<solve_function>"
    code = compile(code, filename, "exec")
    exec(code)
    return locals()[k], factorable
