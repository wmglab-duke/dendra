from .compilers.ast import factorize_linear_in_v
from .compilers.compile_f import convert_func

from axonml.helpers import logger


implicit_equation_template = """
def {k}(self, v):
    gtot_ = {gtot}
    irev = {irev}
    i = gtot_ * (v - irev)
    {assign_to_buffer}
    return i, gtot_
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


def build_current_eq(mechanism, k, assign=False):
    try:
        gtot, irev = factorize_linear_in_v(mechanism.__class__, method=k)
        code = build_implicit_equation(k, gtot, irev, assign)
    except Exception as e:
        raise e
        logger.warning(f"Could not factorize {k} in {mechanism.__class__.__name__}.")
        code = convert_func(getattr(mechanism.__class__, k), assign)
    filename = "<solve_function>"
    code = compile(code, filename, "exec")
    exec(code)
    return locals()[k]