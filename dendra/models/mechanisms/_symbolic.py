import torch  # noqa: F401 - provided to dynamically generated current functions

from dendra.helpers import DEBUG, logger
from dendra.utils.dynamic_compilation import compile_generated_function

from .compilers.ast import EXPECTED_FACTORIZATION_ERRORS, linear_conductance_in_v

implicit_equation_template = """
def {k}(self, v):
    gtot_ = {gtot}
    i = self.{k}(v)
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
    if v.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "Numerical current differentiation supports only torch.float32 and "
            "torch.float64 voltage tensors; use an analytic current/conductance "
            "pair for lower-precision dtypes."
        )
    i = self.{k}(v)
    rel_step = torch.finfo(v.dtype).eps ** (1.0 / 3.0)
    step = rel_step * torch.maximum(torch.abs(v), torch.ones_like(v))
    i_plus = self.{k}(v + step)
    i_minus = self.{k}(v - step)
    {assign_to_buffer}
    return i, (i_plus - i_minus) / (2.0 * step)
"""


analytic_equation_template = """
def {k}(self, v):
    i, g = self.{k}_with_conductance(v)
    {assign_to_buffer}
    return i, g
"""


current_tot_template = """
def {k}_tot(self, v):
{body}
"""


def _annotate_conductance_path(function, mode, fallback_reason=None):
    """Attach stable diagnostics to a generated or user-authored current pair."""
    function._dendra_conductance_mode = mode
    function._dendra_conductance_fallback_reason = fallback_reason
    return function


def build_implicit_equation(current, gtot, assign):
    if assign:
        assign_to_buffer = f"self.{current}_ = i"
    else:
        assign_to_buffer = ""
    return implicit_equation_template.format(
        k=current,
        gtot=gtot,
        assign_to_buffer=assign_to_buffer,
    )


def build_analytic_equation(current, assign):
    if assign:
        assign_to_buffer = f"self.{current}_ = i"
    else:
        assign_to_buffer = ""
    return analytic_equation_template.format(
        k=current,
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
    if k in mechanism._explicit:
        # EXPLICIT is a solver declaration: evaluate the authored current on
        # the RHS and contribute no conductance term, even if the expression is
        # affine or an analytic/numerical derivative was also supplied.
        code = build_unfactorable_equation(k, assign)
        if DEBUG > 0:
            logger.info(f"Generated code for {k}:\n{code}")
        return (
            _annotate_conductance_path(
                compile_generated_function(
                    code,
                    func_name=k,
                    filename_prefix="dendra.mechanisms.explicit",
                    global_ns=globals(),
                ),
                "explicit",
            ),
            False,
        )
    if hasattr(mechanism, f"{k}_with_conductance"):
        # Mechanism.__init__ binds the returned function to ``mechanism`` below.
        # Fetch the descriptor from the class so we return an unbound function;
        # returning the instance attribute here would bind ``self`` twice.
        if not assign:
            return (
                _annotate_conductance_path(
                    getattr(mechanism.__class__, f"{k}_with_conductance"),
                    "analytic",
                ),
                True,
            )

        # SAVE currents must still update their mirrored ``<current>_`` buffer.
        # Wrap the exact pair rather than returning it directly so analytic and
        # generated current paths share the same SAVE contract.
        code = build_analytic_equation(k, assign=True)
        if DEBUG > 0:
            logger.info(f"Generated code for {k}:\n{code}")
        return (
            _annotate_conductance_path(
                compile_generated_function(
                    code,
                    func_name=k,
                    filename_prefix="dendra.mechanisms.analytic",
                    global_ns=globals(),
                ),
                "analytic",
            ),
            True,
        )
    if k in mechanism._numerical:
        code = build_numerical_equation(k, assign)
        factorable = True
        if DEBUG > 0:
            logger.info(f"Generated code for {k}:\n{code}")
        return (
            _annotate_conductance_path(
                compile_generated_function(
                    code,
                    func_name=k,
                    filename_prefix="dendra.mechanisms.numerical",
                    global_ns=globals(),
                ),
                "numerical-declared",
            ),
            factorable,
        )
    fallback_reason = None
    try:
        gtot = linear_conductance_in_v(mechanism.__class__, method=k)
        code = build_implicit_equation(k, gtot, assign)
        factorable = True
        filename_prefix = "dendra.mechanisms.implicit"
        mode = "symbolic"
    except EXPECTED_FACTORIZATION_ERRORS as exc:
        code = build_numerical_equation(k, assign)
        factorable = True
        filename_prefix = "dendra.mechanisms.numerical"
        mode = "numerical-fallback"
        fallback_reason = f"{type(exc).__name__}: {exc}"
    if DEBUG > 0:
        logger.info(f"Generated code for {k}:\n{code}")
    return (
        _annotate_conductance_path(
            compile_generated_function(
                code,
                func_name=k,
                filename_prefix=filename_prefix,
                global_ns=globals(),
            ),
            mode,
            fallback_reason,
        ),
        factorable,
    )
