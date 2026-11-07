import re

import torch  # noqa: F401 - provided to dynamically generated current functions

from dendra.helpers import DEBUG, logger
from dendra.utils.dynamic_compilation import compile_generated_function

from ._numerical import (  # noqa: F401 - used by generated numerical functions
    validate_declared_numerical_current,
)
from .compilers.ast import (
    EXPECTED_FACTORIZATION_ERRORS,
    affine_coefficients_in_v,
    linear_conductance_in_v,
)


class UnsafeAutomaticNumericalFallbackError(RuntimeError):
    """Raised when exact current conductance analysis cannot be completed.

    Numerical differentiation is sound for Dendra's local conductance term only
    when the authored current is deterministic, side-effect free, and pointwise
    in voltage.  A failed symbolic analysis does not establish those properties,
    so silently switching to simultaneous finite differences is unsafe.
    """


def _raise_unsafe_automatic_numerical_fallback(mechanism, current, cause):
    mechanism_name = type(mechanism).__qualname__
    reason = f"{type(cause).__name__}: {cause}"
    raise UnsafeAutomaticNumericalFallbackError(
        f"Could not derive an exact conductance for current {current!r} on "
        f"{mechanism_name}: {reason}. Automatic numerical differentiation is "
        "unsafe because it assumes a deterministic, side-effect-free, pointwise "
        "current; coupled currents otherwise produce a directional derivative "
        "instead of the required Jacobian diagonal. Mechanism currents must be "
        "pointwise; model genuinely coupled voltage dependence outside local "
        "current assembly. For a pointwise current, rewrite it as a supported "
        "affine expression, provide an exact analytic "
        f"{current}_with_conductance(self, v), or explicitly declare "
        f"Mechanism.NUMERICAL({current!r}) to assert that contract."
    ) from cause


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
            f"Numerical current differentiation for current {k!r} on "
            f"{{type(self).__qualname__}} supports only torch.float32 and "
            "torch.float64 voltage tensors; use an analytic current/conductance "
            "pair for lower-precision dtypes."
        )
    if not torch.compiler.is_compiling():
        validate_declared_numerical_current(self, {k!r}, v)
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


def _is_affine_current(mechanism, current):
    """Return whether an affine assertion or source analysis authorizes factoring."""
    if current in mechanism._affine:
        return True
    try:
        linear_conductance_in_v(mechanism.__class__, method=current)
    except EXPECTED_FACTORIZATION_ERRORS:
        return False
    return True


def _broadcast_affine_value(value, voltage):
    """Expand a coefficient before arithmetic with a hidden mapped scalar."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=voltage.dtype, device=voltage.device)
    return torch.broadcast_to(value, voltage.shape)


def affine_hook_name(cls, current):
    """Find a coefficient hook without inheriting it past a current override.

    ``i_affine_terms`` and ``i_with_conductance`` are alternate definitions of
    a current. A more-derived ordinary current/pair must not accidentally use
    coefficients inherited from an older definition. Generated ordinary-current
    adapters do not constitute an authored ownership boundary.
    """
    hook = f"{current}_affine_terms"
    for base in cls.__mro__:
        namespace = base.__dict__
        if hook in namespace:
            return hook if callable(namespace[hook]) else None
        if f"{current}_with_conductance" in namespace:
            return None
        function = namespace.get(current)
        if function is not None and not getattr(
            function, "_dendra_affine_current_adapter", False
        ):
            return None
    return None


def current_contract_owner(cls, current):
    """Owner of the effective authored current, pair, or coefficient definition."""
    for base in cls.__mro__:
        for name in (current, f"{current}_with_conductance", f"{current}_affine_terms"):
            function = base.__dict__.get(name)
            if function is not None and not getattr(
                function, "_dendra_affine_current_adapter", False
            ):
                return base
    return None


def build_affine_current_adapter(current):
    """Synthesize the public current for a coefficient-only authoring API."""
    code = (
        f"def {current}(self, v):\n"
        f"    g, j = self.{current}_affine_terms(v)\n"
        "    return g * v - j\n"
    )
    function = compile_generated_function(
        code,
        func_name=current,
        filename_prefix="dendra.mechanisms.affine_current",
        global_ns=globals(),
    )
    function._dendra_affine_current_adapter = True
    return function


def build_affine_pair_adapter(current, assign, *, authored_current):
    """Use authored coefficients in ordinary current/conductance assembly."""
    lines = [
        f"def {current}(self, v):",
        f"    g, j = self.{current}_affine_terms(v)",
        f"    i = self.{current}(v)" if authored_current else "    i = g * v - j",
    ]
    if assign:
        lines.append(f"    self.{current}_ = i")
    lines.append("    return i, g")
    return compile_generated_function(
        "\n".join(lines) + "\n",
        func_name=current,
        filename_prefix="dendra.mechanisms.affine_pair",
        global_ns=globals(),
    )


def build_affine_pair_coefficients(current):
    """Adapt an asserted affine current's existing exact/numerical pair.

    This preserves the supplied derivative and current arithmetic; authors can
    optionally provide ``i_affine_terms`` to avoid calculating ``I`` here.
    """
    code = (
        f"def _affine_{current}(self, v):\n"
        f"    i, g = self.{current}_with_g(v)\n"
        "    g = _broadcast_affine_value(g, v)\n"
        "    return g, g * v - i\n"
    )
    return compile_generated_function(
        code,
        func_name=f"_affine_{current}",
        filename_prefix="dendra.mechanisms.affine_pair_coefficients",
        global_ns=globals(),
    )


def build_affine_coefficients(mechanism, current):
    """Generate a private zero-conductance-safe ``(G,J)`` symbolic evaluator.

    Symbolic extraction is an optional arithmetic optimization, not a second
    author-facing eligibility contract. Authored hooks and asserted affine
    analytic/numerical pairs use their own coefficient evaluators.
    Each live field is expanded before the first binary arithmetic operation,
    including parameter-only and empty hidden vmap lanes.
    """
    conductance, source = affine_coefficients_in_v(type(mechanism), method=current)
    fields = tuple(
        sorted(set(re.findall(r"\bself\.(\w+)\b", conductance + " " + source)))
    )
    names = {name: f"__affine_{index}" for index, name in enumerate(fields)}

    def substitute(expression):
        return re.sub(r"\bself\.(\w+)\b", lambda match: names[match[1]], expression)

    lines = [f"def _affine_{current}(self, v):"]
    lines.extend(
        f"    {names[name]} = _broadcast_affine_value(self.{name}, v)"
        for name in fields
    )
    lines.extend(
        (
            f"    g = {substitute(conductance)}",
            f"    j = {substitute(source)}",
            "    return _broadcast_affine_value(g, v), _broadcast_affine_value(j, v)",
        )
    )
    function = compile_generated_function(
        "\n".join(lines) + "\n",
        func_name=f"_affine_{current}",
        filename_prefix="dendra.mechanisms.affine_coefficients",
        global_ns=globals(),
    )
    return function, fields


def _assert_affine_close(actual, expected, voltage, message, *, scale=None):
    actual = _broadcast_affine_value(actual, voltage)
    expected = _broadcast_affine_value(expected, voltage)
    # Cancellation in G*v-I and finite-difference conductances require a
    # diagnostic tolerance above machine epsilon. These are bounded probes,
    # not a proof that arbitrary user code is affine everywhere.
    tolerance = 2.0e-4 if voltage.dtype == torch.float32 else 1.0e-7
    absolute = 2.0e-7 if voltage.dtype == torch.float32 else 2.0e-10
    magnitude = torch.maximum(actual.abs(), expected.abs())
    if scale is not None:
        magnitude = torch.maximum(
            magnitude, _broadcast_affine_value(scale, voltage).abs()
        )
    valid = torch.isfinite(actual) & torch.isfinite(expected)
    valid = valid & ((actual - expected).abs() <= absolute + tolerance * magnitude)
    torch._assert_async(torch.all(valid), message)


def _validate_affine_coefficient_metadata(value, voltage, label):
    if torch.is_tensor(value):
        if value.dtype != voltage.dtype or value.device != voltage.device:
            raise RuntimeError(
                label + "coefficient tensor must have the voltage dtype and device"
            )
        try:
            shape = torch.broadcast_shapes(value.shape, voltage.shape)
        except RuntimeError as exc:
            raise RuntimeError(
                label + "coefficient shape must broadcast to the voltage shape"
            ) from exc
        if shape != voltage.shape:
            raise RuntimeError(
                label + "coefficient shape must broadcast to the voltage shape"
            )
    elif type(value) not in (int, float):
        raise RuntimeError(label + "coefficient must be a tensor or numeric scalar")


def validate_affine_current(mechanism, current, voltage, conductance, source):
    """Check frozen-state affine coefficients without advancing gates/ASSIGNED.

    Supports eager execution and ``torch.compile``. Runtime tensor assertions
    have no PyTorch vmap rule; diagnostic validation must precede vmap rather
    than being silently omitted inside it. Normal auto/off assembly supports
    the usual transforms.
    """
    if not torch.compiler.is_compiling():
        tensors = (
            voltage,
            conductance,
            source,
            *mechanism.parameters(),
            *mechanism.buffers(),
        )
        if any(
            torch.is_tensor(value) and torch._C._functorch.is_batchedtensor(value)
            for value in tensors
        ):
            raise RuntimeError(
                "MEMBRANE_ASSEMBLY='validate' does not support vmap; validate "
                "representative unbatched inputs first, then use 'auto' or 'off'."
            )
    label = f"Invalid AFFINE current {current!r} on {type(mechanism).__qualname__}: "
    _validate_affine_coefficient_metadata(conductance, voltage, label + "G ")
    _validate_affine_coefficient_metadata(source, voltage, label + "J ")
    evaluator = getattr(mechanism, mechanism._affine_evaluators[current])
    # Repeated input checks catch inconsistent hooks. Voltage probes leave all
    # mechanism fields fixed: affinity is a solver-stage contract, not a claim
    # about the full voltage-dependent gate-update dynamics.
    pattern = (
        torch.arange(voltage.numel(), device=voltage.device, dtype=voltage.dtype)
        .remainder(3.0)
        .reshape(voltage.shape)
        - 1.0
    )
    for probe in (voltage, voltage - 1.0, voltage + 1.0, voltage + pattern):
        g_probe, j_probe = evaluator(probe)
        _validate_affine_coefficient_metadata(g_probe, voltage, label + "G ")
        _validate_affine_coefficient_metadata(j_probe, voltage, label + "J ")
        _assert_affine_close(
            g_probe,
            conductance,
            voltage,
            label + "G depends on voltage or is inconsistent",
        )
        cancellation_scale = conductance * probe
        _assert_affine_close(
            j_probe,
            source,
            voltage,
            label + "J depends on voltage or is inconsistent",
            scale=cancellation_scale,
        )
        actual = getattr(mechanism, current)(probe)
        _assert_affine_close(
            actual,
            conductance * probe - source,
            voltage,
            label + "coefficients disagree with authored current",
            scale=cancellation_scale,
        )
        paired_current, paired_g = getattr(mechanism, f"{current}_with_g")(probe)
        _assert_affine_close(
            paired_current,
            actual,
            voltage,
            label + "current/conductance pair disagrees with authored current",
            scale=cancellation_scale,
        )
        _assert_affine_close(
            paired_g,
            conductance,
            voltage,
            label + "coefficients disagree with supplied conductance",
        )


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
        factorable = _is_affine_current(mechanism, k)
        # Mechanism.__init__ binds the returned function to ``mechanism`` below.
        # Fetch the descriptor from the class so we return an unbound function;
        # returning the instance attribute here would bind ``self`` twice.
        if not assign:
            return (
                _annotate_conductance_path(
                    getattr(mechanism.__class__, f"{k}_with_conductance"),
                    "analytic",
                ),
                factorable,
            )

        # SAVE_CURRENT mirrors must still receive the evaluated current.
        # Wrap the exact pair rather than returning it directly so analytic and
        # generated current paths share the same saved-current contract.
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
            factorable,
        )
    if affine_hook_name(type(mechanism), k) is not None:
        authored_current = not getattr(
            getattr(type(mechanism), k), "_dendra_affine_current_adapter", False
        )
        return (
            _annotate_conductance_path(
                build_affine_pair_adapter(k, assign, authored_current=authored_current),
                "affine",
            ),
            _is_affine_current(mechanism, k),
        )
    if k in mechanism._numerical:
        code = build_numerical_equation(k, assign)
        factorable = _is_affine_current(mechanism, k)
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
        _raise_unsafe_automatic_numerical_fallback(mechanism, k, exc)
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
